# =============================================================================
# EXPERIMENTAL - NOT in either paper.
# Run with:  --mode exp_echo_srd          (formerly: EchoSRD_with_revision)
# -----------------------------------------------------------------------------
# "Echo"/Alternating SRD: builds TWO decay curves from the same family with
# different alpha (alpha_odd / alpha_even) and alternates them on odd/even
# epochs, decaying down to a keep_floor fraction, then a final full-data revision.
#
# IMPORTANT: despite living next to the ADD variants historically, this is NOT
# an ADD variant. It is fully DETERMINISTIC - there is no Delta_t feedback
# signal and no accept/reject step. The oscillation is pre-scheduled, which is
# precisely the "fixed schedule" property ADD sets out to remove.
#
# SUBSET SAMPLING (paradigm B): _build_subset_loader(), randperm + SubsetRandomSampler.
# EFFECTIVE EPOCHS: total_bp / data_size, printed as "Effective epochs (float)".
#
# start_revision convention here is INVERTED relative to pdd/: <= 0 means
# "auto-place the revision phase at the end" (epochs - revision_epochs).
# =============================================================================
"""
EchoSRD_with_revision.py
Alternating SRD with TWO schedules (odd/even) + keep floor + final revision
--------------------------------------------------------------------------

Key behavior:
- Build TWO per-epoch subset-size schedules using the SAME decay function family
  but different alpha parameters:
    odd epochs  (1,3,5,...)  -> alpha_odd
    even epochs (2,4,6,...)  -> alpha_even

- Schedules are scaled to:
    start = full dataset (N)
    end   = keep_floor * N   (at the END of progressive phase)
  so we do NOT clamp early and "plateau" for many epochs.

- Each epoch draws a RANDOM SUBSET from the FULL dataset (without replacement).
  (Samples may reappear across epochs; that is intended.)

- Final revision:
  If start_revision <= 0, we automatically do "paper style" final revision:
      start_revision = epochs - revision_epochs
  so the last revision_epochs use full data.

Includes:
- plots, effective epochs, samples used per epoch, memory log
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

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

# ======================================================
# === Hyperparameters (edit here) ======================
# ======================================================

# Decay function family (shared by odd/even curves)
# Options:
#   "linear", "geometric",
#   "power_law", "exponential", "log", "inverse_linear", "sigmoid_complement"
DECAY_FUNC = "inverse_linear"

# Two alpha parameters (shape parameters for the chosen decay family)
# For geometric SRD:
#   smaller alpha -> steeper (faster drop)
#   larger alpha  -> less steep (slower drop)
ALPHA_EVEN  = 0.975   # steeper curve (odd epochs)
ALPHA_ODD = 0.5   # less-steep curve (even epochs)

# Keep-floor fraction: schedules decay down to this by end of progressive phase
KEEP_FLOOR = 0.30   # e.g. 10% of dataset

# Final revision behavior (paper style): last N epochs use full dataset
REVISION_FINAL_EPOCH = True
REVISION_EPOCHS = 1

# Optimizer / LR
LR = 3e-4
LR_GAMMA = 0.98

# DataLoader
NUM_WORKERS = 4


# ======================================================
# === Config ===========================================
# ======================================================
@dataclass
class EchoSRDConfig:
    seed: int = 42
    decay_func: str = DECAY_FUNC
    alpha_odd: float = ALPHA_ODD
    alpha_even: float = ALPHA_EVEN
    keep_floor: float = KEEP_FLOOR
    revision_flag: bool = REVISION_FINAL_EPOCH
    revision_epochs: int = REVISION_EPOCHS


# ======================================================
# === Decay curves (raw -> scaled to [N_floor, N]) ======
# ======================================================
def _raw_curve(kind: str, E: int, alpha: float) -> np.ndarray:
    """
    Returns a raw, generally-decreasing array of length E.
    We will normalize + scale it afterward.

    Notes:
    - We use x starting at 1 to keep shapes reasonable.
    - Some curves won't start at 1; we'll normalize by max anyway.
    """
    if E <= 0:
        return np.ones(1, dtype=float)

    x = np.arange(1, E + 1, dtype=float)

    if kind == "linear":
        # alpha not used; pure linear slope in time
        return np.linspace(1.0, 0.0, E, dtype=float)

    if kind == "geometric":
        # classic SRD raw curve: alpha^(t-1)
        a = float(np.clip(alpha, 1e-8, 1.0))
        return (a ** (x - 1.0)).astype(float)

    if kind == "power_law":
        a = max(alpha, 1e-8)
        return (1.0 / (x ** a)).astype(float)

    if kind == "exponential":
        a = max(alpha, 1e-8)
        return np.exp(-a * (x - 1.0)).astype(float)  # starts at 1.0

    if kind == "log":
        # need alpha > ~1 to avoid log small / weird scaling
        a = max(alpha, 1.0001)
        return (1.0 / np.log(x + a)).astype(float)

    if kind == "inverse_linear":
        a = max(alpha, 0.0)
        return (1.0 / (x + a)).astype(float)

    if kind == "sigmoid_complement":
        a = max(alpha, 1e-8)
        # decreasing to 0
        return (1.0 - 1.0 / (1.0 + np.exp(-a * x))).astype(float)

    raise ValueError(f"Unknown decay_func '{kind}'")


def _scale_curve_to_sizes(raw: np.ndarray, N: int, keep_floor: float) -> np.ndarray:
    """
    Normalize raw curve to [0,1] then map to integer subset sizes in [N_floor, N]
    such that:
      size[0]   = N
      size[-1]  = N_floor
    """
    raw = np.asarray(raw, dtype=float)
    E = raw.shape[0]

    keep_floor = float(keep_floor)
    keep_floor = max(min(keep_floor, 1.0), 0.0)
    N_floor = int(round(keep_floor * float(N)))
    N_floor = max(1, min(N_floor, N))

    # Normalize raw to have max=1 (shape only)
    rmax = float(np.max(raw))
    if not np.isfinite(rmax) or rmax <= 0.0:
        # fallback to linear if raw is degenerate
        y = np.linspace(1.0, 0.0, E, dtype=float)
    else:
        y0 = raw / rmax

        # Map so that y_norm[0]=1 and y_norm[-1]=0
        y_start = float(y0[0])
        y_end = float(y0[-1])
        if not np.isfinite(y_start) or not np.isfinite(y_end) or abs(y_start - y_end) < 1e-12:
            y = np.linspace(1.0, 0.0, E, dtype=float)
        else:
            y = (y0 - y_end) / (y_start - y_end)
            y = np.clip(y, 0.0, 1.0)

    # Convert to integer sizes
    sizes = (N_floor + np.round((N - N_floor) * y)).astype(int)

    # Force endpoints exactly
    sizes[0] = N
    sizes[-1] = N_floor

    # Ensure strictly decreasing (prevents “stagnation” from integer rounding)
    for i in range(1, E):
        if sizes[i] >= sizes[i - 1]:
            sizes[i] = max(sizes[i - 1] - 1, N_floor)

    return sizes


# ======================================================
# === Trainer ==========================================
# ======================================================
class EchoSRD:
    def __init__(self, model_name, model, train_loader, test_loader, device, epochs, save_path, threshold):
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = int(epochs)
        self.save_path = save_path
        self.threshold = threshold

        self.dataset = train_loader.dataset
        self.data_size = len(self.dataset)

        # mirror your SA loader setup
        self.batch_size = train_loader.batch_size
        self.num_workers = NUM_WORKERS

    def _build_subset_loader(self, subset_size: int, seed: int) -> Tuple[DataLoader, List[int]]:
        """
        Random subset from full dataset (no replacement within epoch).
        Samples may reappear across epochs (intended).
        """
        N = self.data_size
        subset_size = int(min(max(1, subset_size), N))

        g = torch.Generator()
        g.manual_seed(int(seed))
        idx = torch.randperm(N, generator=g)[:subset_size]
        idx_list = idx.tolist()

        sampler = SubsetRandomSampler(idx_list)
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )
        return loader, idx_list

    @torch.no_grad()
    def _eval(self) -> Tuple[float, float]:
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()

        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.model(inputs)
            loss_sum += float(criterion(outputs, labels).item())
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total += int(labels.size(0))

        acc = correct / max(total, 1)
        loss = loss_sum / max(len(self.test_loader), 1)
        return round(acc, 4), round(loss, 4)

    def train_with_echo_srd(self, cfg: EchoSRDConfig, start_revision: int = 0, task: Optional[str] = None):
        """
        Alternating-SRD training loop with per-epoch subset loader.

        If start_revision <= 0:
          - auto-set start_revision to "final revision epochs" (paper style)
        else:
          - use provided start_revision

        task is unused (kept for signature compatibility).
        """
        save_path = self.save_path
        self.model.to(self.device)

        # Validate
        if not (0.0 < cfg.keep_floor <= 1.0):
            raise ValueError("keep_floor must be in (0,1].")

        if cfg.decay_func == "geometric":
            if not (0.0 < cfg.alpha_odd <= 1.0) or not (0.0 < cfg.alpha_even <= 1.0):
                raise ValueError("For geometric, alpha_odd/alpha_even must be in (0,1].")

        # Determine revision start
        if start_revision is None or int(start_revision) <= 0:
            if cfg.revision_flag and cfg.revision_epochs > 0:
                start_revision = max(self.epochs - int(cfg.revision_epochs), 0)
            else:
                start_revision = self.epochs

        E_prog = int(start_revision)  # number of progressive epochs
        E_prog = max(0, min(E_prog, self.epochs))

        print(f"[ALT-SRD] decay_func={cfg.decay_func} keep_floor={cfg.keep_floor:.3f}")
        print(f"[ALT-SRD] alpha_odd={cfg.alpha_odd} alpha_even={cfg.alpha_even}")
        print(f"[ALT-SRD] progressive_epochs={E_prog}, revision_epochs={self.epochs - E_prog}")

        # Precompute schedules (sizes) for progressive phase only
        if E_prog > 0:
            raw_odd = _raw_curve(cfg.decay_func, E_prog, cfg.alpha_odd)
            raw_even = _raw_curve(cfg.decay_func, E_prog, cfg.alpha_even)

            sizes_odd = _scale_curve_to_sizes(raw_odd, self.data_size, cfg.keep_floor)
            sizes_even = _scale_curve_to_sizes(raw_even, self.data_size, cfg.keep_floor)
        else:
            sizes_odd = np.zeros(1, dtype=int)
            sizes_even = np.zeros(1, dtype=int)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=LR)
        scheduler = StepLR(optimizer, step_size=1, gamma=LR_GAMMA)

        epoch_accuracies = []
        epoch_losses = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        samples_used_per_epoch = []

        total_bp = 0
        start_time = time.time()

        prev_subset_idx = None

        for epoch in range(self.epochs):
            epoch_num = epoch + 1
            epoch_start_time = time.time()

            is_revision = epoch >= E_prog

            # Choose loader
            if is_revision:
                phase = "REVISION"
                loader = self.train_loader
                subset_size = self.data_size
                curve_tag = "FULL"
                alpha_used = None
            else:
                phase = "PROG"
                if (epoch_num % 2) == 1:
                    subset_size = int(sizes_odd[epoch])
                    curve_tag = "ODD"
                    alpha_used = cfg.alpha_odd
                else:
                    subset_size = int(sizes_even[epoch])
                    curve_tag = "EVEN"
                    alpha_used = cfg.alpha_even

                loader, subset_idx = self._build_subset_loader(subset_size, seed=cfg.seed + epoch_num)

                # optional overlap debug
                if prev_subset_idx is not None:
                    s_curr, s_prev = set(subset_idx), set(prev_subset_idx)
                    inter = len(s_curr & s_prev)
                    union = len(s_curr | s_prev)
                    jaccard = inter / union if union > 0 else 0.0
                    print(f"[DATA] Epoch {epoch_num}: Jaccard(prev_subset)={jaccard:.4f}")
                else:
                    print(f"[DATA] Epoch {epoch_num}: Jaccard(prev_subset)=N/A (first epoch)")
                prev_subset_idx = subset_idx

            print(f"\nEpoch [{epoch_num}/{self.epochs}] [{phase}] curve={curve_tag} subset={subset_size}")

            if alpha_used is not None:
                keep_frac = subset_size / float(self.data_size)
                print(f"[ALT-SRD] alpha={alpha_used} keep_frac={keep_frac:.4f}")

            # Train
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in tqdm(loader, desc="Training", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                preds = outputs.argmax(dim=1)
                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))
                total_bp += int(labels.size(0))

            epoch_loss = running_loss / max(len(loader), 1)
            epoch_acc = correct / max(total, 1)

            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_acc)
            samples_used_per_epoch.append(int(total))

            # Eval
            test_acc, val_loss = self._eval()
            epoch_test_accuracies.append(test_acc)
            epoch_test_losses.append(val_loss)

            # keep your existing StepLR call style (consistent with your other scripts)
            scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start_time
            time_per_epoch.append(epoch_time)

            print(f"TrainAcc={epoch_acc:.4f} TrainLoss={epoch_loss:.4f} | TestAcc={test_acc:.4f} TestLoss={val_loss:.4f}")
            print(f"Samples used this epoch: {total}")
            print(f"Epoch time: {epoch_time:.2f}s")

        end_time = time.time()
        log_memory(start_time, end_time)

        eff_epochs = total_bp / float(self.data_size)
        print(f"\nTotal backprop samples: {total_bp}")
        print(f"Effective epochs (float): {eff_epochs:.4f}")

        tag = f"_AltSRD_{cfg.decay_func}_odd{cfg.alpha_odd}_even{cfg.alpha_even}_floor{cfg.keep_floor}"
        plot_accuracy_time_multi(
            model_name=self.model_name + tag,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path,
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name + tag,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path,
        )

        return self.model, int(total_bp)
