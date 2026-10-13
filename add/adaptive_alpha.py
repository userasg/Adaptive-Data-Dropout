# =============================================================================
# ADAPTIVE DATA DROPOUT (ADD) - VARIANT 1 : Adaptive-alpha
# Paper: ADD S3.2 "Variant 1: Adaptive Decay (Adaptive-alpha)" -> paper/ADD.pdf
# Run with:  --mode add_adaptive_alpha        (formerly: saRevisionSimple)
# Result-file tag: "<model>_<threshold>_SA_simple"
# -----------------------------------------------------------------------------
# |D_t| = f(t; alpha_t) * N   where f is one of the five PDD Appendix C decay
# curves and alpha_t is updated online from the training-accuracy feedback
# signal  Delta_t = A_t - A_{t-1}.
#
# ACCEPT / REJECT (see the "---- SA update ----" block in train_with_sa):
#   proposal   a' = clip(alpha +/- alpha_step, 0.1, 20.0)       random direction
#   Delta > 0  -> accept                                        (direct improvement)
#   Delta <= 0 -> Metropolis: accept if rand() < exp(Delta / T)
#   reject     -> alpha -= abs(N(0, 3 * alpha_step))            asymmetric spike,
#                 ALWAYS loosens the schedule => more data next epoch. This is
#                 ADD Figure 3's "Noise added to alpha" box and produces the
#                 reheating spikes seen in ADD Figure 6.
#
# NOTE: T here is a pure Metropolis temperature (acceptance probability only),
# NOT a keep-fraction. That is adaptive_keep_fraction.py's job.
#
# SUBSET SAMPLING (paradigm B): _build_subset_loader() draws |D_t| indices from
# the FULL dataset with torch.randperm (without replacement) and wraps them in a
# SubsetRandomSampler, i.e. a genuinely smaller epoch-level DataLoader. Reseeded
# each epoch as cfg.seed + epoch, so draws are independent across epochs (ADD S3.4).
#
# EFFECTIVE EPOCHS: total_bp accumulates labels.size(0) per batch; printed at the
# end as "Effective epochs (float)" = total_bp / data_size. Returns (model, total_bp).
#
# KNOWN CONFOUND (documented inline in train_with_sa): uses epoch-based StepLR,
# so a shrinking subset means the LR decays after fewer optimiser steps than the
# baseline. adaptive_keep_fraction.py fixes this; this file deliberately does not.
# =============================================================================
"""
saRevisionSimple.py
Simulated Annealing–driven dynamic dropout scheduler (Simple)
-------------------------------------------------------------
Implements function-based SMRD approximation + per-epoch dataset reduction
with simulated annealing tuning of α.

SA Simple is a streamlined version of SA Revision that focuses on the core SA mechanism:
- α controls the SRD decay curve steepness (higher α = more aggressive dropout).
- Each epoch a step of ±SA_ALPHA_STEP is proposed at random.
- ACCEPT  → α takes the proposed value.
- REJECT  → intense downward spike: α -= abs(N(0, step*3)) which forces α lower
             (less aggressive dropout, more samples) as an exploration burst.
             The asymmetric force means a reject always loosens the schedule,
             promoting recovery before the next cooling attempt.
"""

import math
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable
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
#global hyperparameters for SA (edit here only)
SA_DECAY_KIND = "inverse_linear"   # "log", "inverse_linear", "power_law", "exponential", "sigmoid_complement"
SA_ALPHA_INIT = 2.0
SA_ALPHA_STEP = 1.0
SA_ALPHA_MIN = 0.10
SA_ALPHA_MAX = 20.0
SA_T0 = 1.0
SA_TFINAL = 1e-2
SA_COOLING = "geometric"# "linear" or "geometric"
REVISION_FINAL_EPOCH = True# triggers final epoch(s) using full dataset
REVISION_EPOCHS = 1
SA_LR = 3e-4
SA_GAMMA = 0.98
NUM_WORKERS = 4


#dataclass for config
@dataclass
class SAConfig:
    seed: int = 42
    decay_kind: str = SA_DECAY_KIND
    alpha_init: float = SA_ALPHA_INIT
    alpha_step: float = SA_ALPHA_STEP
    T0: float = SA_T0
    Tfinal: float = SA_TFINAL
    cooling: str = SA_COOLING
    revision_epochs: int = REVISION_EPOCHS
    revision_flag: bool = REVISION_FINAL_EPOCH


#decay functions (SRD curve family)
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

DECAY_FUNCS: Dict[str, Callable] = {
    "power_law":          power_law,
    "exponential":        exponential,
    "log":                logarithmic,
    "inverse_linear":     inverse_linear,
    "sigmoid_complement": sigmoid_complement,
}


#SimAnnealDropoutSimple class
class SimAnnealDropoutSimple:
    def __init__(self, model_name, model, train_loader, test_loader, device,
                 epochs, save_path, threshold):
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
        N = len(self.dataset)
        subset_size = min(max(1, subset_size), N)
        g = torch.Generator()
        g.manual_seed(seed if seed is not None else torch.randint(0, 2**31 - 1, (1,)).item())
        idx = torch.randperm(N, generator=g)[:subset_size]
        sampler = SubsetRandomSampler(idx.tolist())
        return DataLoader(
            self.dataset, batch_size=self.batch_size, sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()
        for inputs, labels in self.test_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs  = self.model(inputs)
            loss_sum += criterion(outputs, labels).item()
            correct  += (outputs.argmax(dim=1) == labels).sum().item()
            total    += labels.size(0)
        return correct / total, loss_sum / len(self.test_loader)

    def train_with_sa(self, cfg: SAConfig):
        decay_fn = DECAY_FUNCS[cfg.decay_kind]
        alpha = cfg.alpha_init
        T = cfg.T0
        optimizer = optim.AdamW(self.model.parameters(), lr=SA_LR)

        # LR SCHEDULE: epoch-based (step_size=1 → decay once per epoch).
        # Confound acknowledged: as K_t shrinks, each epoch has fewer batches,
        # so LR decays after fewer optimizer steps than baseline.
        # Accepted here for SA simplicity. See gaRevisionFinal.py for the
        # step-based fix that removes this confound.
        scheduler = StepLR(optimizer, step_size=1, gamma=SA_GAMMA)

        # GRADIENT SCALE: reduction='mean' is safe here because SubsetSampler
        # fills every batch to batch_size before the DataLoader is called.
        # No in-batch masking → no hidden per-sample gradient scaling.
        criterion = nn.CrossEntropyLoss()

        rng = np.random.default_rng(cfg.seed)
        total_bp = 0
        prev_acc = 0.00   # 2dp baseline

        epoch_losses, epoch_accuracies           = [], []
        epoch_test_accuracies, epoch_test_losses = [], []
        samples_used_per_epoch, time_per_epoch   = [], []

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # ---- dataset size for this epoch ----
            is_revision = cfg.revision_flag and (epoch > self.epochs - cfg.revision_epochs)
            subset_size = self.data_size if is_revision else \
                          decay_fn(epoch, self.epochs, self.data_size, alpha)
            subset_size = max(subset_size, self.batch_size)
            loader = self._build_subset_loader(subset_size, seed=cfg.seed + epoch)

            print(f"\nEpoch [{epoch}/{self.epochs}] α={alpha:.2f} T={T:.4f} subset={subset_size}")
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0

            for inputs, labels in tqdm(loader, desc="Training", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
                total_bp += labels.size(0)

            # round to 2dp so all SA comparisons are on equal footing
            train_acc = round(correct / total, 2)
            epoch_loss = running_loss / len(loader)
            samples_used_per_epoch.append(subset_size)
            epoch_accuracies.append(train_acc)
            epoch_losses.append(epoch_loss)

            test_acc, val_loss = self._eval()
            scheduler.step(val_loss)
            epoch_test_accuracies.append(test_acc)
            epoch_test_losses.append(val_loss)

            print(f"TrainAcc={train_acc:.2f}  TrainLoss={epoch_loss:.4f} | "
                  f"TestAcc={test_acc:.4f}  TestLoss={val_loss:.4f}")

            # ---- SA update ----
            if not is_revision and epoch < self.epochs:
                delta = round(train_acc - prev_acc, 2)# 2dp delta
                a_prime = round(float(np.clip(
                    alpha + rng.choice([-1, 1]) * cfg.alpha_step,
                    SA_ALPHA_MIN, SA_ALPHA_MAX)), 2)

                accept = False
                reason = ""
                r, boltz = "-", "-"# shown as dash when Metropolis not reached

                if delta > 0.00:
                    # direct improvement — no Metropolis check needed
                    accept = True
                    reason = "ACCEPT (improvement)"
                else:
                    # Metropolis criterion — all at 2dp for fair comparison
                    r = round(float(rng.random()), 2)
                    boltz = round(math.exp(delta / max(T, 1e-12)), 2)
                    if r < boltz:
                        accept = True
                        reason = "ACCEPT (Metropolis)"
                    else:
                        reason = "REJECT → mutate"

                print(f"[SA] δ={delta:+.2f}  r={r}  boltz={boltz}  T={T:.4f}  → {reason}")

                if accept:
                    alpha = a_prime
                    print(f"[SA]   proposed α={alpha:.2f} accepted")
                else:
                    # REJECT: intense asymmetric spike — always lowers alpha
                    # (lower alpha = less aggressive dropout = more samples next epoch)
                    noise = float(rng.normal(0, cfg.alpha_step * 3.0))
                    alpha_old = alpha
                    alpha = round(float(np.clip(alpha - abs(noise), SA_ALPHA_MIN, SA_ALPHA_MAX)), 2)
                    print(f"[SA]   spike: α {alpha_old:.2f} → {alpha:.2f}  "
                          f"(−{abs(noise):.2f}, more samples next epoch)")

                # ---- cooling schedule ----
                if cfg.cooling == "geometric":
                    ratio = (cfg.Tfinal / cfg.T0) ** (1.0 / max(self.epochs - 1, 1))
                    T = max(T * ratio, cfg.Tfinal)
                else:
                    T = max(T - (cfg.T0 - cfg.Tfinal) / self.epochs, cfg.Tfinal)

            prev_acc = train_acc
            epoch_time = time.time() - epoch_start
            time_per_epoch.append(epoch_time)
            print(f"Epoch time: {epoch_time:.2f}s")

        end_time = time.time()
        log_memory(start_time, end_time)

        eff_epochs = total_bp / self.data_size
        print(f"\nTotal backprop samples: {total_bp}")
        print(f"Effective epochs (float): {eff_epochs:.3f}")

        plot_accuracy_time_multi(
            model_name=self.model_name + "_SA_simple",
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=self.save_path,
            data_file=self.save_path,
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name + "_SA_simple",
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=0.0,
            save_path=self.save_path,
            data_file=self.save_path,
        )

        return self.model, int(total_bp)
