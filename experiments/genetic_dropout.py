# =============================================================================
# EXPERIMENTAL - NOT in either paper.
# Run with:  --mode exp_genetic                      (formerly: gaDropout)
# -----------------------------------------------------------------------------
# Genetic algorithm over PER-SAMPLE survival scores. K_t still follows a PDD SRD
# decay curve, but WHICH samples are kept is decided by a GA rather than by
# confidence or uniform randomness. See the file docstring below for the full
# chromosome / fitness / selection / mutation design.
#
# SUBSET SAMPLING: softmax roulette-wheel selection over survival scores, then
# SubsetRandomSampler (paradigm B).
# EFFECTIVE EPOCHS: total_bp / data_size.
# =============================================================================
"""
gaDropout.py
Genetic Algorithm with per-sample survival scores for PDD.
GA:
  Chromosome  = individual training sample  (gene = survival score s_i ∈ R)
  Population  = all N training samples simultaneously
  Fitness     = per-sample cross-entropy loss attributed within each batch
  Selection   = softmax roulette wheel on scores → choose K_t samples each epoch
  Mutation    = annealed Gaussian noise on non-elite scores
  Crossover   = EMA score mixing (temporal memory between generations)
  K_t         = follows a PDD SRD decay curve (same 5-schedule family as SA)
  Revision    = final epoch(s) train on full dataset D_0

Objective Score:
  High CE loss → score rises → sample is more likely to be selected next epoch.
  This retains hard/informative examples as training progresses, while easy
  samples (low loss) gradually lose score and get deprioritised.
  Mutation + forgetting prevent any sample from being permanently dropped,
  keeping selection stochastic rather than purely deterministic.

Gradient-scale fix:
  Uses reduction='none' to get per-sample losses for score updates, then
  .mean() for backprop. Because SubsetSampler selects K_t samples BEFORE
  building the DataLoader, every retained batch is a proper full-size batch.
  reduction='mean' is therefore consistent — no hidden per-sample gradient
  scaling as K_t shrinks. The concern only arises with in-batch masking;
  it does NOT apply to the SubsetSampler approach used here.

  If switch to in-batch masking, use:
      loss = criterion_sum(outputs, labels_retained) / original_batch_size

LR schedule fix:
  Uses step-based LR decay: scheduler.step() is called after every
  optimizer.step(), and StepLR is initialised with
      step_size = ceil(N / batch_size)   # = one effective full epoch of steps
  This means LR decays at the same rate in terms of optimizer steps regardless
  of how small K_t gets. With epoch-based scheduling (step_size=1), later
  epochs would have fewer batches → LR decays after fewer optimizer steps →
  hidden faster LR decay that could confound PDD results.

Convergence:
  Score entropy H reported each epoch. High H ≈ uniform (random-like, SMRD).
  Low H ≈ concentrated (few samples dominate). Tracks whether the GA
  converges to a structured selection or stays stochastic.
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm

from utils import (
    log_memory,
    plot_accuracy_time_multi,
    plot_accuracy_time_multi_test,
)

# global hyperparameters for GA
GA_DECAY_KIND     = "inverse_linear"
GA_ALPHA          = 1.5              # curve steepness

# score evolution (GA operators)
SCORE_LR          = 0.20             # EMA rate: how fast scores update from fitness
FORGETTING_FACTOR = 0.98             # per-epoch score decay for unselected samples
MUTATION_STD_INIT = 0.10             # initial Gaussian mutation std (anneals to 0)
ELITISM_FRAC      = 0.20             # top fraction of scores protected from mutation
SCORE_MIN         = 0.01
SCORE_MAX         = 2.00

# selection pressure (softmax temperature, anneals upward → sharper over time)
SEL_TEMP_INIT     = 1.00
SEL_TEMP_FINAL    = 5.00
REVISION_FINAL_EPOCH = True
REVISION_EPOCHS      = 1
GA_LR             = 3e-4
GA_GAMMA          = 0.98
NUM_WORKERS       = 4


# decay functions
def power_law(epoch, T, data_size, alpha):
    x = np.arange(1, T + 1)
    y = 1 / (x ** alpha)
    y_scaled = (y / y.max()) * data_size
    return int(y_scaled[epoch - 1])

def exponential(epoch, T, data_size, alpha):
    x = np.arange(1, T + 1)
    y = np.exp(-alpha * x)
    y_scaled = (y / y.max()) * data_size
    return int(y_scaled[epoch - 1])

def logarithmic(epoch, T, data_size, alpha):
    a = max(alpha, 1.0001)
    x = np.arange(1, T + 1)
    y = 1 / np.log(x + a)
    y_scaled = (y / y.max()) * data_size
    return int(y_scaled[epoch - 1])

def inverse_linear(epoch, T, data_size, alpha):
    x = np.arange(1, T + 1)
    y = 1 / (x + alpha)
    y_scaled = (y / y.max()) * data_size
    return int(y_scaled[epoch - 1])

def sigmoid_complement(epoch, T, data_size, alpha):
    x = np.arange(1, T + 1)
    y = 1 - 1 / (1 + np.exp(-alpha * x))
    y_scaled = (y / y.max()) * data_size
    return int(y_scaled[epoch - 1])

DECAY_FUNCS = {
    "power_law":          power_law,
    "exponential":        exponential,
    "log":                logarithmic,
    "inverse_linear":     inverse_linear,
    "sigmoid_complement": sigmoid_complement,
}


# indexed dataset wrapper
class _IndexedDataset(torch.utils.data.Dataset):
    """Wraps any (x, y) dataset to also return the global sample index.

    DataLoader with SubsetRandomSampler(indices) calls dataset[i] for each
    i in the sampler, so the returned idx is the global dataset index.
    This lets us map per-sample CE losses back to scores[N] without a
    separate index-tracking mechanism.
    """
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx


# GaDropout class
class GaDropout:
    def __init__(self, model_name, model, train_loader, test_loader, device,
                 epochs, save_path, threshold, seed=42):
        self.model_name  = model_name
        self.model       = model
        self.test_loader = test_loader
        self.device      = device
        self.epochs      = epochs
        self.save_path   = save_path
        self.batch_size  = train_loader.batch_size
        self.num_workers = NUM_WORKERS

        base_dataset    = train_loader.dataset
        self.data_size  = len(base_dataset)
        self.indexed_ds = _IndexedDataset(base_dataset)

        # precompute K_t schedule for all epochs
        decay_fn         = DECAY_FUNCS[GA_DECAY_KIND]
        self.K_schedule  = []
        for e in range(1, epochs + 1):
            if REVISION_FINAL_EPOCH and e > epochs - REVISION_EPOCHS:
                self.K_schedule.append(self.data_size)  # full dataset for revision
            else:
                K_t = max(self.batch_size, decay_fn(e, epochs, self.data_size, GA_ALPHA))
                self.K_schedule.append(K_t)

        print(f"[GA] K_t schedule: {self.K_schedule[0]} → {self.K_schedule[-1]} "
              f"(revision from epoch {epochs - REVISION_EPOCHS + 1})")

        # per-sample survival scores — chromosome gene for each training example
        rng_init       = np.random.default_rng(seed)
        self.scores    = rng_init.uniform(0.5, 1.0, size=self.data_size).astype(np.float32)
        self.rng       = np.random.default_rng(seed + 1)
        torch.manual_seed(seed)

    def _build_subset_loader(self, idx_array, seed=None):
        """DataLoader over a specific set of global dataset indices."""
        sampler = SubsetRandomSampler(idx_array.tolist())
        return DataLoader(
            self.indexed_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

    def _select_samples(self, epoch):
        """Roulette-wheel selection: draw K_t samples ∝ softmax(scores * T_sel).

        Selection temperature anneals from SEL_TEMP_INIT → SEL_TEMP_FINAL,
        so early epochs have soft (near-random) selection and later epochs
        apply stronger pressure toward high-score samples.
        """
        K_t    = self.K_schedule[epoch - 1]
        p_frac = (epoch - 1) / max(self.epochs - 1, 1)
        T_sel  = SEL_TEMP_INIT + (SEL_TEMP_FINAL - SEL_TEMP_INIT) * p_frac

        logits = self.scores * T_sel
        logits = logits - logits.max()   # numerical stability
        probs  = np.exp(logits)
        probs /= probs.sum()

        selected = self.rng.choice(self.data_size, K_t, replace=False, p=probs)
        return selected

    def _update_scores(self, selected_idx, per_sample_ce):
        """GA fitness update: EMA for selected samples, forgetting for unselected.

        Score semantics: high CE loss → high score → more likely selected next.
        Forgetting (decay) for unselected samples lets dormant examples
        recover over time, preventing permanent exclusion (exploration).
        This is the 'crossover' mechanism: temporal EMA mixes information
        from the current generation into the historical score state.
        """
        sel_mask = np.zeros(self.data_size, dtype=bool)
        sel_mask[selected_idx] = True

        # unselected: gradual forgetting so they can re-enter selection
        self.scores[~sel_mask] *= FORGETTING_FACTOR

        # selected: EMA toward their observed CE loss this epoch
        self.scores[sel_mask] = (
            (1.0 - SCORE_LR) * self.scores[sel_mask] +
            SCORE_LR * per_sample_ce[sel_mask]
        )

        np.clip(self.scores, SCORE_MIN, SCORE_MAX, out=self.scores)

    def _mutate_scores(self, epoch):
        """Annealed Gaussian mutation with elitism protection.

        Mutation std anneals from MUTATION_STD_INIT → 0 over training so
        the population explores broadly early and converges late.
        Elite samples (top ELITISM_FRAC by score) are not mutated — they
        are the 'survivors' carried over to the next generation unchanged.
        """
        p       = (epoch - 1) / max(self.epochs - 1, 1)
        mut_std = MUTATION_STD_INIT * (1.0 - p) ** 2   # quadratic anneal to 0

        if mut_std < 1e-6:
            return

        # protect top ELITISM_FRAC from noise
        elite_thresh = np.percentile(self.scores, (1.0 - ELITISM_FRAC) * 100.0)
        non_elite    = self.scores < elite_thresh

        noise = self.rng.normal(0, mut_std, size=self.data_size).astype(np.float32)
        self.scores[non_elite] += noise[non_elite]
        np.clip(self.scores, SCORE_MIN, SCORE_MAX, out=self.scores)

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total, correct, loss_sum = 0, 0, 0.0
        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs  = self.model(inputs)
            loss_sum += criterion(outputs, labels).item()
            correct  += (outputs.argmax(dim=1) == labels).sum().item()
            total    += labels.size(0)
        return correct / total, loss_sum / len(self.test_loader)

    def train_with_genetic(self):
        # GRADIENT SCALE FIX: reduction='none' gives per-sample CE for score
        # updates. .mean() in the loop is the backprop loss. SubsetSampler fills
        # batches to batch_size before the DataLoader, so mean reduction is
        # consistent across epochs — no hidden gradient scaling as K_t shrinks.
        criterion = nn.CrossEntropyLoss(reduction='none')

        optimizer = optim.AdamW(self.model.parameters(), lr=GA_LR)

        # LR SCHEDULE FIX: step-based decay instead of epoch-based.
        # steps_per_decay = ceil(N / batch_size) = one "effective full epoch"
        # of optimizer steps. scheduler.step() is called after every
        # optimizer.step() so LR decays at a fixed rate in optimizer-step
        # units regardless of K_t. Epoch-based (step_size=1) would decay
        # faster in later epochs when K_t is small and there are fewer batches.
        steps_per_decay = math.ceil(self.data_size / self.batch_size)
        scheduler = StepLR(optimizer, step_size=steps_per_decay, gamma=GA_GAMMA)
        print(f"[GA] Step-based LR: decay every {steps_per_decay} optimizer steps "
              f"(γ={GA_GAMMA}, initial lr={GA_LR})")

        total_bp   = 0
        epoch_losses, epoch_accuracies           = [], []
        epoch_test_accuracies, epoch_test_losses = [], []
        samples_used_per_epoch, time_per_epoch   = [], []

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            is_revision = REVISION_FINAL_EPOCH and (epoch > self.epochs - REVISION_EPOCHS)

            # ---- GA selection (one generation) ----
            if is_revision:
                selected_idx = np.arange(self.data_size)   # full dataset for revision
            else:
                selected_idx = self._select_samples(epoch)

            K_t    = len(selected_idx)
            loader = self._build_subset_loader(selected_idx, seed=epoch)

            print(f"\nEpoch [{epoch}/{self.epochs}] K_t={K_t}/{self.data_size}  "
                  f"score_mean={self.scores.mean():.3f}  score_std={self.scores.std():.3f}")

            # ---- training ----
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            # accumulate per-sample CE losses for score update after the epoch
            per_sample_ce = np.zeros(self.data_size, dtype=np.float32)

            for inputs, labels, global_idx in tqdm(loader, desc="Training", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs  = self.model(inputs)
                ps_loss  = criterion(outputs, labels)   # [B] per-sample CE
                loss     = ps_loss.mean()               # standard mean for backprop
                loss.backward()
                optimizer.step()
                scheduler.step()   # step-based: counts optimizer steps, not epochs

                running_loss += loss.item()
                correct      += (outputs.argmax(1) == labels).sum().item()
                total        += labels.size(0)
                total_bp     += labels.size(0)

                # store per-sample losses (detached) for GA fitness update
                cpu_idx = global_idx.cpu().numpy()
                per_sample_ce[cpu_idx] = ps_loss.detach().cpu().numpy()

            train_acc  = correct / total
            epoch_loss = running_loss / len(loader)
            samples_used_per_epoch.append(K_t)
            epoch_accuracies.append(train_acc)
            epoch_losses.append(epoch_loss)

            # ---- evaluation ----
            # scheduler.step() is NOT called here — it runs inside the batch
            # loop (step-based), so no epoch-level call needed or wanted.
            test_acc, val_loss = self._eval()
            epoch_test_accuracies.append(test_acc)
            epoch_test_losses.append(val_loss)

            print(f"TrainAcc={train_acc:.4f}  TrainLoss={epoch_loss:.4f} | "
                  f"TestAcc={test_acc:.4f}  TestLoss={val_loss:.4f}")

            # ---- GA score update + mutation (skip during revision) ----
            if not is_revision:
                self._update_scores(selected_idx, per_sample_ce)
                self._mutate_scores(epoch)

                # convergence signal: entropy of normalised score distribution
                # high H ≈ uniform selection (SMRD-like); low H ≈ concentrated
                p_norm  = self.scores / self.scores.sum()
                entropy = -float(np.sum(p_norm * np.log(p_norm + 1e-12)))
                max_ent = math.log(self.data_size)
                print(f"[GA] Score entropy H={entropy:.3f} / {max_ent:.1f} "
                      f"({100*entropy/max_ent:.1f}% of uniform)")

            epoch_time = time.time() - epoch_start
            time_per_epoch.append(epoch_time)
            print(f"Epoch time: {epoch_time:.2f}s")

        end_time = time.time()
        log_memory(start_time, end_time)

        eff_epochs = total_bp / self.data_size
        print(f"\nTotal backprop samples: {total_bp}")
        print(f"Effective epochs (float): {eff_epochs:.3f}")

        plot_accuracy_time_multi(
            model_name=self.model_name + "_GA_dropout",
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=self.save_path,
            data_file=self.save_path,
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name + "_GA_dropout",
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=0.0,
            save_path=self.save_path,
            data_file=self.save_path,
        )

        return self.model, int(total_bp)
