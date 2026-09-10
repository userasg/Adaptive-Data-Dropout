# SA Dropout Notes — SARevision & SARevisionSimple

---

## 1. Base Dropout Functions

Both `SARevision.py` and `saRevisionSimple.py` share the same five decay functions. Each function $\hat{f}(x; \alpha)$ maps epoch index $x$ to a sample count, then normalises so the maximum output equals the full dataset size $N$. The final epoch always uses the full dataset.

$$
f(x) = \begin{cases} \hat{f}(x; \alpha) & \text{if } x < T \\ N & \text{if } x = T \end{cases}
$$

### Function Definitions

| # | Name | Formula $\hat{f}(x;\alpha)$ | Behaviour |
|---|---|---|---|
| 1 | Power-Law | $\dfrac{1}{x^\alpha}$ | Very aggressive early drop; tail is long and slow. Highly sensitive to $\alpha$. Higher $\alpha$ → steeper cliff at epoch 1. |
| 2 | Exponential | $e^{-\alpha x}$ | Uniform geometric decay at rate $\alpha$. Most aggressive overall — drops steadily each epoch with no slow tail. |
| 3 | Logarithmic | $\dfrac{1}{\log(x + \alpha)}$ | Slow, gentle decay. The best SMRD approximation per the paper. Stays near full dataset for longer before gradual reduction. |
| 4 | Inverse Linear | $\dfrac{1}{x + \alpha}$ | Moderate hyperbolic decay. Drops fast early then flattens. Most extreme EE savings (9.4 EE at 50 epochs). |
| 5 | Sigmoid-Complement | $1 - \dfrac{1}{1 + e^{-\alpha x}}$ | S-shaped. Delays significant dropout into mid-training, then drops sharper. Smooth transition around inflection point. |

**Effect of $\alpha$:** Lower $\alpha$ → gentler decay (slower dropout, more data retained). Higher $\alpha$ → more aggressive early-stage data reduction.

### Implementation (shared by both files)

Each function is implemented identically in both files. They receive the epoch index, total epochs, dataset size, and α, compute the full curve over all epochs, then return the sample count for the requested epoch:

```python
# saRevisionSimple.py / SARevision.py (identical)
def logarithmic(epoch, T, data_size, alpha):
    a = max(alpha, 1.0001)          # prevent log(1) = 0 at epoch 0
    x = np.arange(1, T + 1)
    y = 1 / np.log(x + a)
    y_scaled = (y / y.max()) * data_size   # normalise: max output = data_size
    return int(y_scaled[epoch - 1])

def inverse_linear(epoch, T, data_size, alpha):
    x = np.arange(1, T + 1)
    y = 1 / (x + alpha)
    y_scaled = (y / y.max()) * data_size
    return int(y_scaled[epoch - 1])

def exponential(epoch, T, data_size, alpha):
    x = np.arange(1, T + 1)
    y = np.exp(-alpha * x)
    y_scaled = (y / y.max()) * data_size
    return int(y_scaled[epoch - 1])
```

The full curve is always recomputed each call (not cached), so `y.max()` normalisation always anchors epoch 1 to the full dataset size.

### Paper Results — CIFAR-100, EfficientNet-B0, 50 epochs from scratch

| Method | Accuracy (%) | Effective Epochs |
|---|---|---|
| Baseline | 63.65 | 50 |
| Inverse Linear | 63.51 | **9.4** |
| Logarithmic Decay | 68.62 | 19.12 |
| **SMRD 0.3** | **69.26** | 16.76 |
| SMRD 0.7 | 67.44 | 21.8 |
| SRD 0.95 | 68.32 | 24.8 |

Logarithmic decay provides the closest approximation to oracle SMRD and retains near-best accuracy. Inverse Linear achieves maximum EE savings but at an accuracy cost.

---

## 2. Simulated Annealing — Background

SA is a stochastic local search metaheuristic. It accepts worse solutions with decreasing probability to escape local optima. The acceptance probability follows the **Boltzmann criterion**: $P(\Delta, T) = \exp(\Delta / T)$, where $\Delta = f(s') - f(s)$ is the quality change and $T$ is the temperature. As $T \to 0$, only improvements are accepted (greedy). SA is **stochastic** (randomness in accept/reject) and **dynamic** (parameters change on a predetermined cooling schedule, not based on performance history).

### General SA Pseudocode

```
INPUT: T₀, T_final
s₀ ← generateInitialSolution
T  ← T₀
s_best ← s₀ ;  s ← s₀

REPEAT
    s' ← perturbation(s)
    Δ   = f(s') − f(s)
    r  ← random ∈ [0, 1)
    if ( r < P(Δ, T) )  {        // accept s' with Boltzmann probability
        s ← s'
    }
    s_best ← updateBest
    T      ← coolTemperature     // decrease T according to cooling schedule
UNTIL (termination conditions satisfied)

Return s_best
```

---

## 3. SARevisionSimple — SA Update for $\alpha$

**What SA tunes:** $\alpha$, the steepness parameter of the chosen decay function.
Higher $\alpha$ → more aggressive early dropout → fewer training samples.
The search space is $\alpha \in [\alpha_{\min}, \alpha_{\max}]$.

### Subset sampling (both files)

Each epoch, a fresh random subset of size `subset_size` is drawn from the full dataset using `SubsetRandomSampler`. No sample is seen more than once within an epoch, but samples can reappear across epochs:

```python
# saRevisionSimple.py
def _build_subset_loader(self, subset_size, seed=None):
    N = len(self.dataset)
    subset_size = min(max(1, subset_size), N)
    g = torch.Generator()
    g.manual_seed(seed if seed is not None else torch.randint(0, 2**31-1, (1,)).item())
    idx = torch.randperm(N, generator=g)[:subset_size]
    sampler = SubsetRandomSampler(idx.tolist())
    return DataLoader(self.dataset, batch_size=self.batch_size, sampler=sampler,
                      num_workers=self.num_workers,
                      pin_memory=(self.device.type == "cuda"))
```

Seeding with `cfg.seed + epoch` ensures reproducible but epoch-distinct subsets.

### SA update in code

```python
# saRevisionSimple.py — SA update (per epoch)
delta   = round(train_acc - prev_acc, 2)
a_prime = round(float(np.clip(
    alpha + rng.choice([-1, 1]) * cfg.alpha_step,
    SA_ALPHA_MIN, SA_ALPHA_MAX)), 2)

if delta > 0.00:
    alpha = a_prime                                     # direct improvement
else:
    r     = round(float(rng.random()), 2)
    boltz = round(math.exp(delta / max(T, 1e-12)), 2)
    if r < boltz:
        alpha = a_prime                                 # Metropolis accept
    else:
        noise = float(rng.normal(0, cfg.alpha_step * 3.0))
        alpha = round(float(np.clip(alpha - abs(noise),
                                    SA_ALPHA_MIN, SA_ALPHA_MAX)), 2)  # spike ↓

# cooling — always runs regardless of accept/reject
if cfg.cooling == "geometric":
    ratio = (cfg.Tfinal / cfg.T0) ** (1.0 / max(self.epochs - 1, 1))
    T = max(T * ratio, cfg.Tfinal)
else:
    T = max(T - (cfg.T0 - cfg.Tfinal) / self.epochs, cfg.Tfinal)
```

```
╔══════════════════════════════════════════════════════════════════════╗
║  Algorithm 2: SARevisionSimple — SA update for α                    ║
║  Input: α, Δα > 0, T, cooling ∈ {geometric, linear}, ε > 0         ║
╚══════════════════════════════════════════════════════════════════════╝

for each epoch e do

    α' ← clip( α + choice({−1, +1}) · Δα,  α_min, α_max )

    subset_size ← decay_func(e, E, N, α')
    Δ ← round( Acc_train(α') − Acc_train(α),  2dp )

    if Δ > ε  then
    │   α ← α'                              // strict improvement

    else
    │   p ← exp( Δ / max(T, 1e-12) )
    │   draw u ~ U(0, 1)
    │   if u < p  then
    │   │   α ← α'                          // Metropolis accept
    │   else
    │   │   noise ← |N(0, 3·Δα)|
    │   │   α ← clip( α − noise,  α_min, α_max )   // asymmetric spike ↓

    // cooling (every epoch, regardless of accept/reject)
    if cooling == "geometric" then
    │   T ← max( T · (T_final/T₀)^(1/(E−1)),  T_final )
    else
    │   T ← max( T − (T₀ − T_final)/E,  T_final )
```

**Reject behaviour:** On reject, $\alpha$ is forced **downward** by a large Gaussian spike. Lower $\alpha$ means less aggressive dropout → more training data next epoch. This is an asymmetric, one-directional recovery: reject always loosens the schedule, never tightens it. Temperature $T$ still cools regardless of outcome.

---

## 4. SARevision — T-Schedule with Reheating

**What SA tunes:** $T$ directly — the keep-fraction of the full dataset, $T \in [T_{\text{final}}, T_0]$.
Subset size each epoch: $K_e = \text{round}(T \cdot N)$.

The decay function acts as a **proposal trajectory** $T_{\text{base}}[e]$, pre-computed once at init.
On **accept** → $T$ steps forward along the curve.
On **reject** → $T$ **reheats** upward (more data), then re-tracks the curve on the next accept.

### Pre-computation in code

```python
# SARevision.py — _build_base_temperature_schedule()
func = DECAY_FUNCS[cooling]          # e.g. logarithmic
y_vals = []
for e in range(1, E + 1):
    y_vals.append(func(e, E, data_size=1.0, alpha=1.0))   # α fixed at 1 — shape only
y_vals = np.array(y_vals, dtype=float)
y_min, y_max = float(y_vals.min()), float(y_vals.max())

def shaped_T(e: int) -> float:
    y    = float(y_vals[e - 1])
    norm = (y - y_min) / (y_max - y_min)          # squeeze to [0, 1]
    val  = cfg.Tfinal + norm * (cfg.T0 - cfg.Tfinal)   # stretch to [Tfinal, T0]
    return round(val, 4)
```

### SA update in code

```python
# SARevision.py — SA update (per epoch)
delta = round(train_acc - prev_acc, 2)

if delta > cfg.delta_threshold:
    T = round(T_base(epoch + 1), 4)                   # ACCEPT: advance curve

else:
    boltz_prob = math.exp(cfg.boltz_scalar * delta / T)
    r_value    = rng.random()

    if r_value < boltz_prob:
        T = round(T_base(epoch + 1), 4)               # ACCEPT (Metropolis)
    else:
        T = round(min(cfg.T0, T * SA_REHEAT_MULT), 4) # REJECT: reheat ↑
```

```
╔══════════════════════════════════════════════════════════════════════╗
║  Algorithm 3: SARevision — T-schedule with reheating                ║
╚══════════════════════════════════════════════════════════════════════╝

// PRE-COMPUTATION (once, before training)
for e = 1 to E do
│   y[e] ← decay_func(e, E, data_size=1.0, α=1.0)   // α fixed at 1
norm[e] ← ( y[e] − min(y) ) / ( max(y) − min(y) )
T_base[e] ← T_final + norm[e] · (T₀ − T_final)      // rescale to [T_final, T₀]

// TRAINING LOOP
T ← T₀
for each epoch e do

    K_e ← round( T · N )                             // dataset size this epoch
    train on K_e samples
    Δ ← round( Acc_train(e) − Acc_train(e−1),  2dp )

    if Δ > δ_threshold  then
    │   T ← T_base[e+1]                              // ACCEPT: follow curve

    else
    │   p ← exp( λ · Δ / T )                         // λ = boltz_scalar
    │   draw u ~ U(0, 1)
    │   if u < p  then
    │   │   T ← T_base[e+1]                          // ACCEPT (Metropolis)
    │   else
    │   │   T ← min( T₀,  T · reheat_mult )          // REJECT: reheat ↑
```

**Reheating logic:** On reject, $T$ is multiplied by `reheat_mult` (2.5), snapping it back toward $T_0 = 1.0$ (full dataset). The model gets more data before the next cooling attempt. The more rejects occur, the further $T$ lags behind $T_{\text{base}}$. The decay curve is a target trajectory, not a hard schedule.

### How T should scale with total epochs

The pre-computation normalises the decay curve shape over exactly $E$ epochs, so the curve always spans from $T_0$ down to $T_{\text{final}}$ regardless of epoch count. The **shape** is self-adjusting. However, $T_{\text{final}}$ itself is a fixed floor — and this is where epoch count matters.

**The problem:** With `SA_TFINAL = 0.05` (5% data) and `SA_REHEAT_MULT = 2.5`:
- At 200 epochs, there is enough time to gradually descend the curve and spend meaningful epochs near mid-T before bottoming out.
- At 30 epochs, the curve still reaches 5% by epoch 29 — but because there are few epochs, the model may spend too many of them at very low T (almost no data) with little time to recover before the revision epoch. Any string of rejects also collapses back to T₀, which leaves only a handful of epochs to re-descend the entire curve.

**The fix:** `SA_TFINAL` should be raised for shorter runs, and optionally `SA_REHEAT_MULT` reduced so reheats are less violent:

| Total Epochs | Recommended `SA_TFINAL` | Recommended `SA_REHEAT_MULT` | Rationale |
|---|---|---|---|
| 30 | 0.20 | 1.5 | Can't afford large reheats or very low T floor with only 30 epochs |
| 100 | 0.10 | 2.0 | Moderate: enough epochs to recover from a reheat |
| 200 | 0.05 | 2.5 | Default — current setting; full curve has room to breathe |

Conceptually, $T_{\text{final}}$ sets how deeply into the dropout schedule you commit before the revision epoch. A rough heuristic is:

$$T_{\text{final}} \approx \max\!\left(0.05,\ \frac{5}{\sqrt{E}}\right)$$

which gives ≈0.33 at $E=30$, ≈0.16 at $E=100$, and ≈0.11 at $E=200$ — slightly more conservative than the table above but principled.

**What this looks like in practice for log cooling (`SA_COOLING = "log"`):**

```python
# How T_base is distributed across epochs for different E values
# (log curve, T0=1.0, Tfinal varies)

# E=30,  Tfinal=0.20: T_base spans [0.20, 1.0] across 30 epochs
#   → epoch 1:  T=1.00  (100% data)
#   → epoch 15: T≈0.40  (40% data, curve midpoint)
#   → epoch 29: T=0.20  (20% data before revision)

# E=200, Tfinal=0.05: T_base spans [0.05, 1.0] across 200 epochs
#   → epoch 1:   T=1.00  (100% data)
#   → epoch 100: T≈0.07  (7% data, near floor early due to log shape)
#   → epoch 199: T=0.05  (5% data before revision)
```

With log cooling, the curve descends quickly then flattens — so for 200 epochs the model is already near `Tfinal` by roughly epoch 80-100. Reheats at that late stage snap T all the way back to 1.0, which is a very large jump. For short runs this is unworkable. This is why both `SA_TFINAL` and `SA_REHEAT_MULT` need to be considered together when changing epoch count.

---

## 5. Hyperparameters

### SARevisionSimple (`saRevisionSimple.py`)

| Hyperparameter | Value | Description | Rationale |
|---|---|---|---|
| `SA_DECAY_KIND` | `"inverse_linear"` | Default decay function | Extreme EE savings (9.4 EE per paper); aggressive starting point |
| `SA_ALPHA_INIT` | 2.0 | Initial $\alpha$ | Moderate early dropout; not too aggressive for cold-start |
| `SA_ALPHA_STEP` ($\Delta\alpha$) | 1.0 | Perturbation step size | Allows meaningful jump per epoch; large enough to explore |
| `SA_ALPHA_MIN` | 0.10 | Lower bound on $\alpha$ | Near-zero dropout; prevents complete dataset collapse |
| `SA_ALPHA_MAX` | 20.0 | Upper bound on $\alpha$ | Near-total early elimination; extreme aggression ceiling |
| `SA_T0` | 1.0 | Initial temperature | Standard SA convention |
| `SA_TFINAL` | 0.01 | Final temperature | Near-deterministic at end of training |
| `SA_COOLING` | `"geometric"` | Temperature cooling schedule | Smooth exponential decay; standard SA convention |
| `SA_LR` | 3e-4 | AdamW learning rate | Standard for AdamW on CIFAR |
| `SA_GAMMA` | 0.98 | StepLR decay per epoch | Matches paper's SRD/SMRD LR schedule |
| `REVISION_EPOCHS` | 1 | Final full-dataset epochs | Paper: 1 revision epoch is optimal (Table 15) |

**Search landscape:** The SA explores $\alpha \in [0.1, 20]$ — a 1D continuous space with a single step size. The landscape is unimodal-ish per function (higher $\alpha$ → more dropout → fewer samples → risk of underfitting). Reject spikes provide large explorative jumps downward; accepts provide ±1 refinement.

---

### SARevision (`SARevision.py`)

| Hyperparameter | Value | Description | Rationale |
|---|---|---|---|
| `SA_COOLING` | `"log"` | Shape function for $T_{\text{base}}$ curve | Log decay = best SMRD approximation per paper (Table 21) |
| `SA_T0` | 1.0 | Initial keep-fraction (100% data) | Full dataset at epoch 1 |
| `SA_TFINAL` | 0.05 | Minimum keep-fraction (5% data) | Paper results plateau near 5–10% dataset before revision |
| `SA_DELTA_THRESHOLD` ($\delta$) | 0.005 | Auto-accept threshold | Avoids triggering on floating-point noise |
| `SA_REHEAT_MULT` | 2.5 | Reheating multiplier | Snaps T back near T₀ quickly on poor epochs |
| `SA_BOLTZ_SCALAR` ($\lambda$) | 2.5 | Boltzmann amplifier | $>1$ → more acceptance (less reheating); tunable exploration-exploitation |
| `SA_LR` | 3e-4 | AdamW learning rate | Standard for AdamW on CIFAR |
| `SA_GAMMA` | 0.98 | LR step decay gamma | Step-based (not epoch-based) — see section 6 |
| `REVISION_EPOCHS` | 1 | Final full-dataset epochs | Paper: 1 revision epoch is optimal |

**Search landscape:** The SA explores $T \in [0.05, 1.0]$ — the fraction of the dataset to use. The landscape is determined by accuracy response to dataset size. Reheating ($\times 2.5$) creates a strong bias toward larger datasets on poor epochs. $\lambda = 2.5$ amplifies the Boltzmann probability, favouring acceptance of the cooling proposal.

---

## 6. LR Schedule & Gradient Scale Updates

These changes exist **only in `SARevision.py`**. `saRevisionSimple.py` intentionally omits them for simplicity (acknowledged in docstring).

### Change 1 — Step-based LR decay

**Before (`saRevisionSimple.py`):**
```python
scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
# stepped once per epoch, after evaluation
```

**After (`SARevision.py`):**
```python
steps_per_decay = math.ceil(self.data_size / self.batch_size)
scheduler = StepLR(optimizer, step_size=steps_per_decay, gamma=0.98)
# stepped inside the batch loop — scheduler.step() after optimizer.step()
```

**Why:** With shrinking subsets, later epochs have fewer batches. Epoch-based decay drops LR after fewer optimizer steps when the subset is small — a hidden LR acceleration confounded with the dropout effect. Step-based decay normalises to approximately one full-dataset equivalent of optimizer steps, making LR decay comparable across all epochs regardless of $K_e$.

---

### Change 2 — Gradient scale fix

**Before (`saRevisionSimple.py`):**
```python
criterion = nn.CrossEntropyLoss()           # default: mean reduction
loss = criterion(outputs, labels)
```

**After (`SARevision.py`):**
```python
criterion = nn.CrossEntropyLoss(reduction='sum')
loss = criterion(outputs, labels) / self.batch_size   # normalise by original batch size
```

**Why:** The last mini-batch of an epoch may be smaller than `batch_size`. With mean reduction, each sample in that smaller batch contributes more to the gradient than samples in full batches — gradient scale is inconsistent across batches. Dividing the summed loss by the original `batch_size` keeps gradient magnitude stable regardless of actual batch size. This is particularly relevant when $K_e$ is small and the final batch is often much smaller than `batch_size`.
