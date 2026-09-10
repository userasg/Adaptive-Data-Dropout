"""
SARevision.py
Simulated Annealing–driven dynamic dropout scheduler
- No explicit alpha parameter.
- Each epoch uses a random subset of the *full* dataset (no replacement within
  epoch, but samples may reappear across epochs).
- Temperature T directly controls the fraction of data kept.
- Cooling schedule (linear, geometric, or one of the analytic curves below)
  proposes a lower T; SA acceptance can either:
- accept the cooler T (smaller subset), or
- reject and "reheat" by increasing T (larger subset).
- This variant controls reheating frequency by scaling the Boltzmann exponent.
"""
import math
import time
from dataclasses import dataclass
from typing import Dict, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import numpy as np 

# plotting utils (same as in TrainRevision)
from utils import (
    log_memory,
    plot_accuracy_time_multi,
    plot_accuracy_time_multi_test,
)

#global hyperparameters for SA (edit here only)
# SA_COOLING can be:
# "linear", "geometric",
# or one of: "power_law", "exponential", "log",
# "inverse_linear", "sigmoid_complement"
SA_COOLING = "log"
SA_T0 = 1.0000 # initial keep-fraction (use 100% of dataset)
SA_TFINAL = 0.0500 # final keep-fraction (e.g. 5% of dataset)
SA_DELTA_THRESHOLD = 0.0050 # accuracy improvement threshold to auto-accept cooling
SA_REHEAT_MULT = 2.5000 # how strongly to reheat (T <- min(T0, T * SA_REHEAT_MULT))

#Reheating control: Boltzmann scalar
SA_BOLTZ_SCALAR = 2.5000 #>1 => more acceptance (less reheating), <1 => more reheating
REVISION_FINAL_EPOCH = True # final revision epochs with full dataset
REVISION_EPOCHS = 1 # number of revision epochs
SA_LR = 3e-4
SA_GAMMA = 0.9800 # LR scheduler gamma (StepLR), NOT SA temperature
NUM_WORKERS = 4

@dataclass
class SAConfig:
    seed: int = 42
    T0: float = SA_T0
    Tfinal: float = SA_TFINAL
    cooling: str = SA_COOLING
    revision_epochs: int = REVISION_EPOCHS
    revision_flag: bool = REVISION_FINAL_EPOCH
    delta_threshold: float = SA_DELTA_THRESHOLD
    boltz_scalar: float = SA_BOLTZ_SCALAR


#analytic cooling-shape functions
def power_law(epoch: int, total_epochs: int, data_size: float, alpha: float) -> float:
    x = np.arange(1, total_epochs + 1)
    y = 1.0 / (x ** alpha)
    y_scaled = (y / y.max()) * data_size
    return float(y_scaled[epoch - 1])


def exponential(epoch: int, total_epochs: int, data_size: float, alpha: float) -> float:
    x = np.arange(1, total_epochs + 1)
    y = np.exp(-alpha * x)
    y_scaled = (y / y.max()) * data_size
    return float(y_scaled[epoch - 1])


def logarithmic(epoch: int, total_epochs: int, data_size: float, alpha: float) -> float:
    a = max(alpha, 1.0001)
    x = np.arange(1, total_epochs + 1)
    y = 1.0 / np.log(x + a)
    y_scaled = (y / y.max()) * data_size
    return float(y_scaled[epoch - 1])


def inverse_linear(epoch: int, total_epochs: int, data_size: float, alpha: float) -> float:
    x = np.arange(1, total_epochs + 1)
    y = 1.0 / (x + alpha)
    y_scaled = (y / y.max()) * data_size
    return float(y_scaled[epoch - 1])


def sigmoid_complement(epoch: int, total_epochs: int, data_size: float, alpha: float) -> float:
    x = np.arange(1, total_epochs + 1)
    y = 1.0 - 1.0 / (1.0 + np.exp(-alpha * x))
    y_scaled = (y / y.max()) * data_size
    return float(y_scaled[epoch - 1])


DECAY_FUNCS: Dict[str, Callable] = {
    "power_law": power_law,
    "exponential": exponential,
    "log": logarithmic,
    "inverse_linear": inverse_linear,
    "sigmoid_complement": sigmoid_complement,
}

class SimAnnealDropout:
    def __init__(self, model_name, model, train_loader, test_loader, device, epochs, save_path, threshold):
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.save_path = save_path
        self.dataset = train_loader.dataset
        self.data_size = len(self.dataset)
        self.batch_size = train_loader.batch_size
        self.num_workers = NUM_WORKERS

    def _build_subset_loader(self, subset_size, seed=None):
        """
        Version (1) behaviour:
        - Sample a random subset from the *full* dataset every epoch.
        - Without replacement *within* epoch (SubsetRandomSampler),
          but samples may reappear across epochs.
        Returns:
            loader, idx_list (indices used this epoch).
        """
        N = len(self.dataset)
        subset_size = min(max(1, subset_size), N)
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        else:
            g.manual_seed(torch.randint(0, 2**31 - 1, (1,)).item())
        idx = torch.randperm(N, generator=g)[:subset_size]
        idx_list = idx.tolist()
        sampler = SubsetRandomSampler(idx_list)
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return loader, idx_list

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()
        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss_sum += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        acc = correct / max(total, 1)
        loss = loss_sum / max(len(self.test_loader), 1)
        return round(acc, 4), round(loss, 4)

    def _build_base_temperature_schedule(self, cfg: SAConfig):
        """
        Helper to build the *baseline* temperature schedule T_base(e)
        from epoch index -> keep-fraction, using:
          - "linear"
          - "geometric"
          - or one of the analytic curves in DECAY_FUNCS.
        All schedules are scaled to lie exactly in [cfg.Tfinal, cfg.T0].
        """
        cooling = cfg.cooling
        E = self.epochs

        if cooling == "linear":
            def linear_T(e: int) -> float:
                if E <= 1:
                    return round(cfg.Tfinal, 4)
                frac = (e - 1) / max(E - 1, 1)
                val = cfg.T0 + frac * (cfg.Tfinal - cfg.T0)
                return round(val, 4)

            return linear_T

        if cooling == "geometric":
            if E <= 1:
                ratio = 1.0
            else:
                ratio = (cfg.Tfinal / cfg.T0) ** (1.0 / max(E - 1, 1))

            def geometric_T(e: int) -> float:
                val = cfg.T0 * (ratio ** (e - 1))
                val = max(val, cfg.Tfinal)
                return round(val, 4)

            return geometric_T

        if cooling not in DECAY_FUNCS:
            raise ValueError(f"Unknown cooling schedule '{cooling}'")

        func = DECAY_FUNCS[cooling]
        y_vals = []
        for e in range(1, E + 1):
            y_vals.append(func(e, E, data_size=1.0, alpha=1.0))
        y_vals = np.array(y_vals, dtype=float)
        y_min = float(y_vals.min())
        y_max = float(y_vals.max())

        if y_max == y_min:
            def flat_T(e: int) -> float:
                return round(cfg.Tfinal, 4)
            return flat_T

        def shaped_T(e: int) -> float:
            y = float(y_vals[e - 1])
            norm = (y - y_min) / (y_max - y_min)
            val = cfg.Tfinal + norm * (cfg.T0 - cfg.Tfinal)
            return round(val, 4)

        return shaped_T

    def train_with_sa(self, cfg: SAConfig):
        """
        Main training loop for Version 3 (boltz scalar).
        - Temperature T is interpreted as *keep-fraction* of the dataset.
        - Subset size per epoch = round(T * data_size), min = batch_size.
        - SA uses training accuracy improvement as the fitness of the current
          subset (for that epoch).
        - No explicit alpha; we only adjust T via cooling + reheating.
        """
        T_base = self._build_base_temperature_schedule(cfg)

        optimizer = optim.AdamW(self.model.parameters(), lr=SA_LR)

        # LR SCHEDULE FIX: step-based decay so that LR decays after the same
        # number of *optimizer steps* regardless of how large K_t is.
        # Epoch-based StepLR(step_size=1) would decay after fewer steps when K_t
        # shrinks → hidden faster LR decay that confounds PDD vs baseline comparison.
        # Fix: count gradient steps, decay once per equivalent full-dataset pass.
        steps_per_decay = math.ceil(self.data_size / self.batch_size)
        scheduler = StepLR(optimizer, step_size=steps_per_decay, gamma=SA_GAMMA)

        # GRADIENT SCALE FIX: use reduction='sum' then divide by self.batch_size
        # so that the mean is always over the *original* batch size, not the
        # actual number of samples in the (possibly smaller) last mini-batch.
        # Without this, the final mini-batch of an epoch contributes a higher
        # per-sample gradient weight than earlier full batches.
        criterion = nn.CrossEntropyLoss(reduction='sum')

        rng = np.random.default_rng(cfg.seed)
        total_bp = 0
        prev_acc = 0.0

        epoch_losses, epoch_accuracies = [], []
        epoch_test_accuracies, epoch_test_losses = [], []
        samples_used_per_epoch, time_per_epoch = [], []

        start_time = time.time()
        prev_subset_idx = None

        T = round(cfg.T0, 4)

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.time()

            is_revision = cfg.revision_flag and (epoch > self.epochs - cfg.revision_epochs)

            if is_revision:
                T_epoch = 1.0000
            else:
                T_epoch = round(T, 4)

            subset_size = int(round(T_epoch * float(self.data_size)))
            subset_size = max(subset_size, self.batch_size)
            subset_size = min(subset_size, self.data_size)

            loader, subset_idx = self._build_subset_loader(subset_size, seed=cfg.seed + epoch)

            if prev_subset_idx is not None:
                s_curr = set(subset_idx)
                s_prev = set(prev_subset_idx)
                inter = len(s_curr & s_prev)
                union = len(s_curr | s_prev)
                jaccard = inter / union if union > 0 else 0.0
                jaccard = round(jaccard, 4)
                print(
                    f"[DATA] Epoch {epoch}: Jaccard(prev_subset) = {jaccard:.4f} "
                    f"(|curr|={len(s_curr)}, |prev|={len(s_prev)}, |∩|={inter})"
                )
            else:
                print(f"[DATA] Epoch {epoch}: Jaccard(prev_subset) = N/A (first epoch)")

            prev_subset_idx = subset_idx

            print(f"\nEpoch [{epoch}/{self.epochs}] T_epoch={T_epoch:.4f} subset={subset_size}")
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in tqdm(loader, desc="Training", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels) / self.batch_size  # gradient scale fix: normalise by original batch size
                loss.backward()
                optimizer.step()
                scheduler.step()   # step-based: counts optimizer steps, not epochs
                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
                total_bp += labels.size(0)

            train_acc = correct / max(total, 1)
            epoch_loss = running_loss / max(len(loader), 1)
            train_acc = round(train_acc, 4)
            epoch_loss = round(epoch_loss, 4)

            samples_used_per_epoch.append(subset_size)
            epoch_accuracies.append(train_acc)
            epoch_losses.append(epoch_loss)

            test_acc, val_loss = self._eval()
            # scheduler.step() removed here step-based decay runs inside the batch loop above
            epoch_test_accuracies.append(test_acc)
            epoch_test_losses.append(val_loss)

            print(f"TrainAcc={train_acc:.4f} TestAcc={test_acc:.4f}")
            # ---- SA update ----
            if not is_revision and epoch < self.epochs:
                delta_raw = train_acc - prev_acc
                delta = round(delta_raw, 4)
                accept = False
                reason = "none"

                if delta > cfg.delta_threshold:
                    accept = True
                    reason = "threshold"
                else:
                    r_value = rng.random()

                    boltz_prob_raw = math.exp(
                        cfg.boltz_scalar * delta / T
                    )
                    accept_prob = boltz_prob_raw

                    print(
                        f"[SA][DEBUG] epoch={epoch} delta={delta:+.4f}, "
                        f"T={T:.4f}, r={r_value:.4f}, "
                        f"boltz_prob_raw={boltz_prob_raw:.4f}, "
                        f"accept_prob={accept_prob:.4f}, "
                        f"scalar={cfg.boltz_scalar:.4f}"
                    )

                    if r_value < accept_prob:
                        accept = True
                        reason = "metropolis"
                    else:
                        accept = False
                        reason = "reheat"

                if accept:
                    #temp store old and new T for logging clarity
                    T_old = T
                    #get what t should be at the next epoch according to the base cooling schedule
                    T_new = T_base(epoch + 1)
                    T = round(T_new, 4)
                    print(
                        f"[SA][ACCEPT:{reason}] epoch={epoch} delta={delta:+.4f} "
                        f"T: {T_old:.4f} → {T:.4f}"
                    )
                else:
                    T_old = T
                    T_new = min(cfg.T0, T * SA_REHEAT_MULT)
                    T = round(T_new, 4)
                    print(
                        f"[SA][REHEAT] epoch={epoch} delta={delta:+.4f} "
                        f"T: {T_old:.4f} → {T:.4f}"
                    )

            prev_acc = train_acc
            epoch_time = time.time() - epoch_start_time
            epoch_time = round(epoch_time, 4)
            time_per_epoch.append(epoch_time)
            print(f"Epoch time: {epoch_time:.4f}s")

        end_time = time.time()
        log_memory(start_time, end_time)

        eff_epochs = total_bp / float(self.data_size)
        eff_epochs = round(eff_epochs, 4)
        print(f"\nTotal backprop samples: {total_bp}")
        print(f"Effective epochs (float): {eff_epochs:.4f}")

        plot_accuracy_time_multi(
            model_name=self.model_name + "_SA_v3_scalar",
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=self.save_path,
            data_file=self.save_path,
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name + "_SA_v3_scalar",
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=0.0,
            save_path=self.save_path,
            data_file=self.save_path,
        )

        num_step = int(total_bp)
        return self.model, num_step