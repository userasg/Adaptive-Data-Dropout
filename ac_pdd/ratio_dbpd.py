"""
ratio_dbpd.py - AC-PDD: Adaptive-Composition Progressive Data Dropout
=====================================================================
Source: the AC_PDD working repo (Morgan Jones), itself built on the original
DBPD/SMRD release for "Progressive Data Dropout" (arXiv 2505.22342). Ported
here unchanged in behaviour; only the comments and the enclosing class name
are new.

WHAT AC-PDD IS
--------------
PDD gives us two ends of a spectrum, both already in pdd/pdd_training_variants.py:

    DBPD (pdd_dbpd)   back-props on the HARD samples only - whichever samples
                      currently score below the confidence threshold tau.
    SMRD (pdd_smrd)   back-props on a RANDOM subset whose SIZE is matched to
                      DBPD's schedule, ignoring which samples are hard.

DBPD's size schedule turns out to be most of where the speed-up comes from
(that is SMRD's whole point), but hard-only selection and random selection are
not equally good for accuracy at every stage of training. AC-PDD makes the
choice between them a continuous, scheduled parameter rather than a fork in the
road.

For each batch it keeps DBPD's BUDGET but reshapes that budget's COMPOSITION:

    target_budget       = hard_available            <- identical to DBPD
    natural_hard_frac   = hard_available / batch_size
    target_hard_frac    = (1 - lambda_t) + lambda_t * natural_hard_frac

So with the same number of slots to fill:

    lambda_t = 0  ->  target_hard_frac = 1.0
                      every slot goes to a hard sample. Exactly DBPD.
    lambda_t = 1  ->  target_hard_frac = natural_hard_frac
                      slots are filled in the batch's own hard/easy ratio,
                      i.e. a uniform random draw from the batch in
                      expectation. Exactly SMRD.
    0 < lambda_t < 1  ->  a mixture: mostly-hard with a growing easy share.

lambda_t is then RAMPED across the selection phase by utils.get_lambda_t:

    progress = epoch / (selection_epochs - 1)
    lambda_t = lambda_min + (lambda_max - lambda_min) * progress ** gamma

which is why the method is "adaptive composition": the same budget is spent on
an increasingly random (less hard-biased) sample as training proceeds.
--lambda_min / --lambda_max set the two endpoints and --gamma shapes the curve
between them (gamma > 1 = slow start then fast finish; gamma < 1 = the reverse).

Note that lambda_min=0.0, lambda_max=0.0 reproduces DBPD and
lambda_min=1.0, lambda_max=1.0 reproduces SMRD, so the two PDD baselines are
reachable as special cases of this one method.

THE FLOOR VARIANT
-----------------
`ratio_dbpd` inherits one pathology from DBPD: the budget IS the hard count, so
as the model fits the data the budget collapses. Over a 200-epoch CIFAR run the
selection pool shrinks toward a few hundred of 50,000 samples by around epoch
100 and stays there, so the back half of the run re-memorises a tiny residual
instead of training on anything new.

`ratio_dbpd_floor` adds a lower bound: once hard_available alone would put the
budget under `floor_fraction * batch_size`, the budget is raised to that floor
and the extra slots are filled by the same lambda_t-weighted split - which,
with the hard pool exhausted, is a uniform random draw from the batch's easy
pool. The floor therefore introduces no second selection mechanism; it stops
the existing one from shrinking below a minimum, and tops it up the way SMRD
would.

It is kept as a separate method rather than a flag on `ratio_dbpd` so the
un-floored version remains a clean comparison condition. The two are identical
apart from `floor_fraction` being threaded into `ratio_dbpd_select` and three
extra diagnostic columns in the per-epoch CSV.

A NOTE ON --gamma
-----------------
`--gamma` here is the lambda_t ramp exponent. It is unrelated to the StepLR
`gamma=0.98` learning-rate decay factor used below, which is fixed and is NOT
exposed as a flag. Two different gammas, one shared name - inherited from the
upstream repo and kept so AC_PDD's own commands still run unchanged.

OUTPUTS
-------
Unlike every other variant in this repo, these two methods are instrumented
with the MetricsRecorder from ac_pdd/learning_cycles.py, so a run writes:

    {save_path}_epochs.parquet                        per-epoch metrics
    {save_path}_samples.parquet                       per-sample, per-epoch
    {save_path}_identifiers.parquet                   sample_id -> label
    {save_path}_outcomes.parquet                      final per-sample state
    {save_path}_run_metadata.json                     the run's config
    {save_path}_dynamics.png                          summary plot
    {save_path}_ratio_dbpd[_floor]_epoch_stats.csv    selection diagnostics
    {save_path}_checkpoint.pt                         resume state, deleted on
                                                      clean completion
    {save_path}_test / _test.png                      the legacy accuracy-vs-
                                                      time JSON + plot that
                                                      collect_results.py reads

Reading the parquet files needs pyarrow (pip install pyarrow).

EFFECTIVE EPOCHS
----------------
Both methods accumulate `num_step` at the point of backprop only and return
(model, num_step), matching every other variant, so main.py's
EE = num_step / dataset_size figure is directly comparable to PDD and ADD runs.
"""
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (log_memory, plot_accuracy_time_multi,
                   plot_accuracy_time_multi_test, get_lambda_t)
from ac_pdd.learning_cycles import (MetricsRecorder, IndexedDataset,
                                    per_sample_loss, make_run_id)


# =============================================================================
# Long-tail loss, used only when --task is not "classification".
# Carried over from the upstream file so this module stays self-contained;
# nothing else in this repo defines it.
# =============================================================================
def focal_loss(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


# =============================================================================
# THE SELECTION RULE - this is the whole of AC-PDD's contribution.
# =============================================================================
def ratio_dbpd_select(hard_mask, easy_mask, lambda_t, floor_fraction=0.0):
    """Pick which samples in one batch get back-propped.

    Args:
        hard_mask:  bool tensor, True where the sample is "hard" right now
                    (correct-class probability < tau, or simply misclassified
                    when tau == 0). Computed by the caller.
        easy_mask:  ~hard_mask.
        lambda_t:   composition weight for THIS epoch, from utils.get_lambda_t.
                    0 = fill every slot with a hard sample (DBPD);
                    1 = fill slots in the batch's natural hard/easy ratio
                    (SMRD in expectation).
        floor_fraction: minimum budget as a fraction of batch size. 0.0 (the
                    default, and what `ratio_dbpd` passes) means no floor, so
                    the budget is exactly DBPD's. `ratio_dbpd_floor` passes a
                    positive value.

    Returns (retained_hard_mask, retained_easy_mask, stats), where the two
    masks are disjoint subsets of hard_mask / easy_mask and `stats` carries the
    realised counts for the per-epoch diagnostics CSV.

    How the budget is set and spent:

      1. BUDGET.  target_budget = max(hard_available, floor_count). Without a
         floor this is just hard_available, i.e. DBPD's budget. The budget is
         a COUNT of slots and does not depend on lambda_t at all - lambda_t
         only decides who fills them.

      2. COMPOSITION.  target_hard_fraction interpolates between 1.0 and the
         batch's own hard fraction:
             (1 - lambda_t) + lambda_t * (hard_available / batch_size)
         then target_hard_count = round(budget * that), clamped to what is
         actually available, and the rest of the budget goes to easy samples.

      3. DRAW.  torch.randperm over each pool, so which hard samples and which
         easy samples get picked is uniformly random within its pool. Only the
         hard/easy PROPORTION is scheduled, never the identity of the samples.

      4. TOP-UP.  If clamping left the budget unfilled (one pool ran dry), the
         shortfall is taken from whatever the other pool has spare. This is
         what makes the floor work: with the hard pool exhausted, the floor's
         extra slots come from a uniform random draw over the easy pool -
         exactly what SMRD would do - rather than from re-picking the same
         shrinking hard residual.
    """
    device = hard_mask.device
    hard_indices = torch.where(hard_mask)[0]
    easy_indices = torch.where(easy_mask)[0]
    hard_available = hard_indices.numel()
    easy_available = easy_indices.numel()
    batch_size = hard_available + easy_available

    floor_count = min(batch_size, int(round(floor_fraction * batch_size)))
    target_budget = max(hard_available, floor_count)
    floor_triggered = target_budget > hard_available
    natural_hard_fraction = hard_available / batch_size if batch_size > 0 else 0.0
    target_hard_fraction = (1.0 - lambda_t) + lambda_t * natural_hard_fraction

    target_hard_count = int(round(target_budget * target_hard_fraction))
    target_hard_count = max(0, min(target_hard_count, hard_available))
    target_easy_count = target_budget - target_hard_count
    target_easy_count = max(0, min(target_easy_count, easy_available))

    hard_perm = torch.randperm(hard_available, device=device) if hard_available > 0 else torch.empty(0, dtype=torch.long, device=device)
    easy_perm = torch.randperm(easy_available, device=device) if easy_available > 0 else torch.empty(0, dtype=torch.long, device=device)

    selected_hard_local = hard_perm[:target_hard_count]
    selected_easy_local = easy_perm[:target_easy_count]

    shortfall = target_budget - (selected_hard_local.numel() + selected_easy_local.numel())
    if shortfall > 0:
        spare_hard = hard_perm[target_hard_count:]
        take = min(shortfall, spare_hard.numel())
        if take > 0:
            selected_hard_local = torch.cat([selected_hard_local, spare_hard[:take]])
            shortfall -= take
        if shortfall > 0:
            spare_easy = easy_perm[target_easy_count:]
            take = min(shortfall, spare_easy.numel())
            if take > 0:
                selected_easy_local = torch.cat([selected_easy_local, spare_easy[:take]])
                shortfall -= take

    selected_hard_indices = hard_indices[selected_hard_local]
    selected_easy_indices = easy_indices[selected_easy_local]

    retained_hard_mask = torch.zeros_like(hard_mask)
    retained_easy_mask = torch.zeros_like(easy_mask)
    if selected_hard_indices.numel() > 0:
        retained_hard_mask[selected_hard_indices] = True
    if selected_easy_indices.numel() > 0:
        retained_easy_mask[selected_easy_indices] = True

    selected_hard_count = int(retained_hard_mask.sum().item())
    selected_easy_count = int(retained_easy_mask.sum().item())
    stats = {
        "raw_hard_available": hard_available,
        "raw_easy_available": easy_available,
        "target_hard_count": target_hard_count,
        "target_easy_count": target_easy_count,
        "selected_hard_count": selected_hard_count,
        "selected_easy_count": selected_easy_count,
        "dbpd_target_budget": target_budget,
        "actual_selected_count": selected_hard_count + selected_easy_count,
        "floor_count": floor_count,
        "floor_triggered": floor_triggered,
    }
    return retained_hard_mask, retained_easy_mask, stats

class RatioDBPD:
    """AC-PDD trainer. Deliberately a separate class from pdd.TrainRevision.

    The two AC-PDD methods are instrumented with MetricsRecorder and support
    mid-run checkpoint/resume, neither of which pdd/pdd_training_variants.py
    has. Grafting that onto the shared PDD class would have changed the
    behaviour of all twelve pdd_* modes, so AC-PDD brings its own loop instead.
    The constructor takes the same first eight arguments as TrainRevision, plus
    run_id / seed / dataset which only the recorder needs.
    """

    def __init__(self, model_name, model, train_loader, test_loader, device,
                 epochs, save_path, threshold,
                 run_id=None, seed=None, dataset=None):
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.save_path = save_path
        self.threshold = threshold
        self.run_id = run_id
        self.seed = seed
        self.dataset = dataset

    # -- resume support -------------------------------------------------------
    # A run writes {save_path}_checkpoint.pt after every epoch and deletes it
    # on clean completion, so re-running the same --save_path after a crash
    # picks up where it stopped. Note the flip side: a *different* run reusing
    # a save_path whose previous run died will silently resume it.
    def _load_checkpoint(self, ckpt_path):

        if not os.path.exists(ckpt_path):
            return None
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        return checkpoint

    def _save_checkpoint(self, ckpt_path, epoch, run_id, optimizer, scheduler, **extra):
        state = {
            "epoch": epoch,
            "run_id": run_id,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }
        state.update(extra)
        torch.save(state, ckpt_path)

    # =========================================================================
    # AC-PDD v1  -  no budget floor                            --mode acpdd_ratio
    # -------------------------------------------------------------------------
    # DBPD's budget (target_budget = hard_available), lambda_t-scheduled
    # composition. Because the budget still tracks the hard count, the
    # selection pool collapses as the model fits - see ratio_dbpd_floor for the
    # variant that bounds it.
    #
    # Signature note: `cls_num_list` is only read when --task is not
    # "classification"; `gamma` is the lambda_t ramp exponent, NOT the StepLR
    # decay factor set a few lines into the body.
    # =========================================================================
    def ratio_dbpd (self, start_revision, task, cls_num_list, lambda_min, lambda_max, gamma=0.5):

        save_path = self.save_path
        self.model.to(self.device)
        if task=='classification':
            criterion = nn.CrossEntropyLoss()
        elif 'longtail':
            train_sampler = None
            idx = self.epochs // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.device)
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(self.device)
        # optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        #as per implementation LR=0.045, they use 16 GPU. https://discuss.pytorch.org/t/training-mobilenet-on-imagenet/174391/6 from this blog
        #we use the idea to divide the learning rate by the number of GPUs.
        # optimizer = optim.RMSprop(self.model.parameters(), weight_decay=0.00004, momentum=0.9, lr=0.0028125)
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        survival_log = defaultdict(list)
        label_log = defaultdict(int)
        start_time = time.time()
        num_step = 0
        samples_used_per_epoch = []

        ckpt_path = f"{save_path}_checkpoint.pt"
        start_epoch = 0
        pending_recorder_state = None

        hard_count_epoch0 = None
        hard_count_prev_epoch = None

        # Ratio-DBPD's own per-epoch selection diagnostics - a separate
        # CSV (same pattern as margin_pdd_epoch_stats.csv) rather than
        # extending MetricsRecorder's shared schema, so DBPD/SMRD/
        # soft_dbpd are untouched. Rewritten in full after every epoch
        # so a crash mid-run still leaves every completed epoch's stats
        # readable; read back here so a resume picks up where it left off.
        ratio_dbpd_stats_path = f"{save_path}_ratio_dbpd_epoch_stats.csv"
        ratio_dbpd_stat_rows = pd.read_csv(ratio_dbpd_stats_path).to_dict("records") if os.path.exists(ratio_dbpd_stats_path) else []

        checkpoint = self._load_checkpoint(ckpt_path)
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            run_id = checkpoint["run_id"]
            start_epoch = checkpoint["epoch"] + 1
            epoch_losses = checkpoint["epoch_losses"]
            epoch_accuracies = checkpoint["epoch_accuracies"]
            epoch_test_accuracies = checkpoint["epoch_test_accuracies"]
            epoch_test_losses = checkpoint["epoch_test_losses"]
            time_per_epoch = checkpoint["time_per_epoch"]
            samples_used_per_epoch = checkpoint["samples_used_per_epoch"]
            num_step = checkpoint["num_step"]
            survival_log = defaultdict(list, checkpoint["survival_log"])
            label_log = defaultdict(int, checkpoint["label_log"])
            pending_recorder_state = checkpoint["recorder_state"]
            hard_count_epoch0 = checkpoint.get("hard_count_epoch0")
            hard_count_prev_epoch = checkpoint.get("hard_count_prev_epoch")
            print(f"[train_with_revision] Resumed from checkpoint: epoch {start_epoch}/{self.epochs} (run_id={run_id})")
        else:
            run_id = self.run_id if self.run_id else make_run_id(self.model_name)

        indexed_train_loader = DataLoader(IndexedDataset(self.train_loader.dataset), batch_size=self.train_loader.batch_size, shuffle=True)
        recorder = MetricsRecorder(
            self.model_name, save_path, run_id=run_id, seed=self.seed, dataset=self.dataset, architecture=self.model_name,
            optimizer_name="AdamW", learning_rate=3e-4, total_epochs=self.epochs, batch_size=self.train_loader.batch_size,
            threshold=self.threshold, start_revision=start_revision, mode="train_with_revision",
        )
        if pending_recorder_state is not None:
            recorder.import_state(pending_recorder_state)
        epoch0_sample_ids, epoch0_labels = [], []

        for epoch in range(start_epoch, self.epochs):
            samples_used = 0
            current_lr = optimizer.param_groups[0]["lr"]
            # hard_fraction is kept only for logging/checkpoint compatibility
            # with earlier runs - it no longer drives selection (see
            # ratio_dbpd_select below, which uses lambda_t and the batch's
            # own natural hard/easy composition instead).
            hard_fraction = 1.0 if hard_count_epoch0 is None else (hard_count_prev_epoch / hard_count_epoch0)
            hard_samples_present = 0
            # Counted directly from what actually got backpropped (see the
            # fix below), not derived by subtracting from samples_used -
            # that subtraction went negative because it summed easy over a
            # different set of batches than samples_used did.
            hard_used_count_this_epoch = 0
            easy_used_count_this_epoch = 0
            lambda_t = get_lambda_t(epoch, self.epochs, lambda_min, lambda_max, gamma, start_revision)

            # Ratio-DBPD selection diagnostics, aggregated from actual
            # per-batch realised counts (see ratio_dbpd_select) - not
            # estimated from probabilities.
            ratio_dbpd_raw_hard_available = 0
            ratio_dbpd_raw_easy_available = 0
            ratio_dbpd_target_hard_count = 0
            ratio_dbpd_target_easy_count = 0
            ratio_dbpd_selected_hard_count = 0
            ratio_dbpd_selected_easy_count = 0
            ratio_dbpd_target_budget = 0
            ratio_dbpd_actual_selected_count = 0

            if epoch < start_revision :
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total_correct = 0
                total_samples = 0
                total = 0
                print(f"Epoch [{epoch+1/self.epochs}]")
                progress_bar = tqdm(enumerate(indexed_train_loader), total=len(indexed_train_loader), desc="Training")

                for batch_idx, (inputs, labels, sample_ids) in progress_bar:
                    batch_start_idx = batch_idx * self.train_loader.batch_size
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        prob = torch.softmax(outputs, dim=1)
                        correct_class = prob[torch.arange(labels.size(0)), labels]
                        sample_losses = per_sample_loss(outputs, labels)

                        if self.threshold == 0:
                            hard_mask = preds != labels
                        else:
                            hard_mask = correct_class < self.threshold
                        easy_mask = ~hard_mask

                        # DBPD's own budget for this batch is however many
                        # samples are hard right now; lambda_t interpolates
                        # the composition of that fixed-size budget between
                        # hard-only (DBPD, lambda_t=0) and the batch's
                        # natural hard/easy proportion (SMRD-like,
                        # lambda_t=1) - see ratio_dbpd_select's docstring.
                        retained_hard_mask, retained_easy_mask, sel_stats = ratio_dbpd_select(hard_mask, easy_mask, lambda_t)

                        hard_samples_present += sel_stats["raw_hard_available"]
                        ratio_dbpd_raw_hard_available += sel_stats["raw_hard_available"]
                        ratio_dbpd_raw_easy_available += sel_stats["raw_easy_available"]
                        ratio_dbpd_target_hard_count += sel_stats["target_hard_count"]
                        ratio_dbpd_target_easy_count += sel_stats["target_easy_count"]
                        ratio_dbpd_selected_hard_count += sel_stats["selected_hard_count"]
                        ratio_dbpd_selected_easy_count += sel_stats["selected_easy_count"]
                        ratio_dbpd_target_budget += sel_stats["dbpd_target_budget"]
                        ratio_dbpd_actual_selected_count += sel_stats["actual_selected_count"]

                        mask = retained_hard_mask | retained_easy_mask

                    recorder.log_batch(epoch, sample_ids.tolist(), labels, preds, outputs, sample_losses, correct_class, mask)
                    if epoch == 0:
                        epoch0_sample_ids.extend(sample_ids.tolist())
                        epoch0_labels.extend(labels.detach().cpu().tolist())

                    if mask.sum().item() < 2:
                        continue

                    # Counted here, not above: a batch that fails the >=2
                    # gate contributes nothing to samples_used, so it must
                    # contribute nothing here either, or hard_used +
                    # easy_used stops summing to samples_used.
                    hard_used_count_this_epoch += retained_hard_mask.sum().item()
                    easy_used_count_this_epoch += retained_easy_mask.sum().item()

                    inputs_misclassified = inputs[mask]
                    labels_misclassified = labels[mask]

                    used_labels = labels_misclassified
                    for label in used_labels.tolist():
                        label_log[int(label)] += 1

                    misclassified_in_batch = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    absolute_indices = (misclassified_in_batch + batch_start_idx).tolist()
                    survival_log[epoch].extend(absolute_indices)

                    optimizer.zero_grad()

                    outputs_misclassified = self.model(inputs_misclassified)
                    loss = criterion(outputs_misclassified, labels_misclassified)
                    num_step+=len(outputs_misclassified)
                    samples_used+=len(outputs_misclassified)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                    progress_bar.set_postfix({"Loss": loss.item()})

                # This epoch's raw (unscaled) hard count becomes next
                # epoch's `hard_count_prev_epoch`, and epoch 0's raw hard
                # count anchors the schedule for every later epoch.
                # Retained for logging/checkpoint compatibility only -
                # selection itself no longer reads these (see
                # ratio_dbpd_select).
                if hard_count_epoch0 is None:
                    hard_count_epoch0 = hard_samples_present
                hard_count_prev_epoch = hard_samples_present

                budget_fill_rate = (ratio_dbpd_actual_selected_count / ratio_dbpd_target_budget) if ratio_dbpd_target_budget > 0 else 1.0
                realised_hard_proportion = (ratio_dbpd_selected_hard_count / ratio_dbpd_actual_selected_count) if ratio_dbpd_actual_selected_count > 0 else 0.0
                realised_easy_proportion = (ratio_dbpd_selected_easy_count / ratio_dbpd_actual_selected_count) if ratio_dbpd_actual_selected_count > 0 else 0.0
                ratio_dbpd_stat_rows.append({
                    "epoch": epoch, "phase": "selection", "lambda_t": lambda_t,
                    "raw_hard_available": ratio_dbpd_raw_hard_available,
                    "raw_easy_available": ratio_dbpd_raw_easy_available,
                    "target_hard_count": ratio_dbpd_target_hard_count,
                    "target_easy_count": ratio_dbpd_target_easy_count,
                    "selected_hard_count": ratio_dbpd_selected_hard_count,
                    "selected_easy_count": ratio_dbpd_selected_easy_count,
                    "dbpd_target_budget": ratio_dbpd_target_budget,
                    "actual_selected_count": ratio_dbpd_actual_selected_count,
                    "budget_fill_rate": budget_fill_rate,
                    "realised_hard_proportion": realised_hard_proportion,
                    "realised_easy_proportion": realised_easy_proportion,
                    # Diagnostic-only, unused by selection - see hard_fraction's comment above.
                    "hard_fraction_diagnostic": hard_fraction,
                })
                pd.DataFrame(ratio_dbpd_stat_rows).to_csv(ratio_dbpd_stats_path, index=False)
                print(f"[ratio_dbpd] Epoch {epoch + 1}/{self.epochs}: "
                      f"hard={ratio_dbpd_selected_hard_count}, easy={ratio_dbpd_selected_easy_count}, "
                      f"total={ratio_dbpd_actual_selected_count} (dbpd_target_budget={ratio_dbpd_target_budget})")

                if epoch == 0:
                    recorder.log_identifiers(epoch0_sample_ids, epoch0_labels)

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct/total_samples if total_samples > 0 else 0
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time-epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)

                accuracy = correct / total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step()
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

                recorder.log_epoch(
                    epoch=epoch,
                    learning_rate=current_lr,
                    threshold=self.threshold,
                    train_loss=epoch_loss,
                    train_accuracy=epoch_accuracy,
                    val_loss=val_loss,
                    val_accuracy=accuracy,
                    samples_used=samples_used,
                    samples_total=total_samples,
                    time_seconds=time_per_epoch[-1],
                    phase="selection",
                    hard_fraction=hard_fraction,
                    lambda_t=lambda_t,
                    # hard_samples_present is the raw (unscaled) count of
                    # hard samples the epoch actually produced; only
                    # `hard_fraction` of those were retained, backfilled
                    # with easy samples up to the same slot count.
                    # hard_used_count_this_epoch and easy_used_count_this_epoch
                    # are each counted directly from the same >=2-gated
                    # batches samples_used comes from (see above), so the
                    # two sum to samples_used exactly - do not replace
                    # either with samples_used minus the other: that
                    # subtraction previously summed easy over every batch
                    # while samples_used only summed over ungated ones,
                    # which went negative once most batches were skipped.
                    hard_available_count=hard_samples_present,
                    easy_available_count=total_samples - hard_samples_present,
                    hard_used_count=hard_used_count_this_epoch,
                    easy_used_count=easy_used_count_this_epoch,
                )

            else:
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total = 0
                hard_count_this_epoch = 0

                print(f"Epoch [{epoch+1/self.epochs}]")
                progress_bar = tqdm(enumerate(indexed_train_loader), total=len(indexed_train_loader), desc="Training")


                for batch_idx, (inputs, labels, sample_ids) in progress_bar:
                    batch_start_idx = batch_idx * self.train_loader.batch_size
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    absolute_indices = list(range(batch_start_idx, batch_start_idx + inputs.size(0)))
                    survival_log[epoch].extend(absolute_indices)
                    used_labels = labels
                    for label in used_labels.tolist():
                        label_log[int(label)] += 1
                    num_step+=len(outputs)
                    samples_used+=len(outputs)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        prob = torch.softmax(outputs, dim=1)
                        correct_class = prob[torch.arange(labels.size(0)), labels]
                        sample_losses = per_sample_loss(outputs, labels)
                        kept_mask = torch.ones_like(labels, dtype=torch.bool)  # revision phase: all samples kept
                        # Revision trains on everyone regardless, but the
                        # underlying hard/easy split (same criterion as the
                        # pre-revision phase) is still worth knowing here -
                        # this reuses the correct_class kept_mask already
                        # needed, at zero extra forward passes.
                        if self.threshold == 0:
                            hard_count_this_epoch += (preds != labels).sum().item()
                        else:
                            hard_count_this_epoch += (correct_class < self.threshold).sum().item()
                        recorder.log_batch(epoch, sample_ids.tolist(), labels, preds, outputs, sample_losses, correct_class, kept_mask)
                        if epoch == 0:
                            epoch0_sample_ids.extend(sample_ids.tolist())
                            epoch0_labels.extend(labels.detach().cpu().tolist())
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                if epoch == 0:
                    recorder.log_identifiers(epoch0_sample_ids, epoch0_labels)

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step()
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

                # No Ratio-DBPD selection runs during revision (full
                # dataset, unweighted, unchanged) - a NaN-filled row keeps
                # the CSV epoch-indexed without implying selection stats
                # that were never computed this epoch.
                ratio_dbpd_stat_rows.append({
                    "epoch": epoch, "phase": "revision", "lambda_t": lambda_t,
                    "raw_hard_available": float("nan"), "raw_easy_available": float("nan"),
                    "target_hard_count": float("nan"), "target_easy_count": float("nan"),
                    "selected_hard_count": float("nan"), "selected_easy_count": float("nan"),
                    "dbpd_target_budget": float("nan"), "actual_selected_count": float("nan"),
                    "budget_fill_rate": float("nan"), "realised_hard_proportion": float("nan"),
                    "realised_easy_proportion": float("nan"), "hard_fraction_diagnostic": hard_fraction,
                })
                pd.DataFrame(ratio_dbpd_stat_rows).to_csv(ratio_dbpd_stats_path, index=False)
                print(f"[ratio_dbpd] Epoch {epoch + 1}/{self.epochs} (revision): "
                      f"hard={hard_count_this_epoch}, easy={total - hard_count_this_epoch}, "
                      f"total={total} (full dataset, no selection)")

                recorder.log_epoch(
                    epoch=epoch,
                    learning_rate=current_lr,
                    threshold=self.threshold,
                    train_loss=epoch_loss,
                    train_accuracy=epoch_accuracy,
                    val_loss=val_loss,
                    val_accuracy=accuracy,
                    samples_used=samples_used,
                    samples_total=total,
                    time_seconds=time_per_epoch[-1],
                    phase="revision",
                    hard_fraction=hard_fraction,
                    lambda_t=lambda_t,
                    # Revision trains on everyone, so used == available for
                    # both hard and easy here.
                    hard_available_count=hard_count_this_epoch,
                    easy_available_count=total - hard_count_this_epoch,
                    hard_used_count=hard_count_this_epoch,
                    easy_used_count=total - hard_count_this_epoch,
                )

            samples_used_per_epoch.append(samples_used)

            # This epoch's samples file is finalized (footer written, readable)
            # before the checkpoint is saved, so a crash on the very next line
            # never leaves an unreadable half-written samples file behind.
            recorder.finalize_epoch_samples()
            self._save_checkpoint(
                ckpt_path, epoch, run_id, optimizer, scheduler,
                epoch_losses=epoch_losses, epoch_accuracies=epoch_accuracies,
                epoch_test_accuracies=epoch_test_accuracies, epoch_test_losses=epoch_test_losses,
                time_per_epoch=time_per_epoch, samples_used_per_epoch=samples_used_per_epoch,
                num_step=num_step, survival_log=dict(survival_log), label_log=dict(label_log),
                recorder_state=recorder.export_state(),
                hard_count_epoch0=hard_count_epoch0, hard_count_prev_epoch=hard_count_prev_epoch,
            )

        self.model.eval()
        final_ids, final_labels, final_preds, final_logits, final_losses, final_probs = [], [], [], [], [], []
        with torch.no_grad():
            for inputs, labels, sample_ids in tqdm(indexed_train_loader, desc="Final outcomes pass"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                prob = torch.softmax(outputs, dim=1)
                correct_class = prob[torch.arange(labels.size(0)), labels]
                sample_losses = per_sample_loss(outputs, labels)
                final_ids.extend(sample_ids.tolist())
                final_labels.append(labels)
                final_preds.append(preds)
                final_logits.append(outputs)
                final_losses.append(sample_losses)
                final_probs.append(correct_class)
        recorder.log_outcomes(
            final_ids, torch.cat(final_labels), torch.cat(final_preds), torch.cat(final_logits),
            torch.cat(final_losses), torch.cat(final_probs),
        )

        recorder.merge_epoch_samples()
        recorder.close()
        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        total_wall_time = end_time - start_time
        print(f"\n Total Wall Time for {self.epochs} epochs: {total_wall_time:.2f} seconds "
            f"({total_wall_time / 60:.2f} minutes)")


        plot_accuracy_time_multi(
        model_name=self.model_name,
        accuracy=epoch_accuracies,
        time_per_epoch=time_per_epoch,
        save_path=save_path,
        data_file=save_path
        )
        plot_accuracy_time_multi_test(
            model_name = self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path
        )

        survival_log_path = os.path.join(os.path.dirname(save_path), "survival_log_eff.json")
        with open(survival_log_path, "w") as f:
            json.dump(dict(survival_log), f, indent=2)
        print(f"Survival log saved to {survival_log_path}")

        label_log_path = os.path.join(os.path.dirname(save_path), "label_log_eff.json")
        with open(label_log_path, "w") as f:
            json.dump(dict(label_log), f, indent=2)
        print(f"Survival log saved to {label_log_path}")

        # Training completed all `epochs` without crashing - no longer need the
        # resume checkpoint, and leaving it behind would make a *later*,
        # unrelated run at the same save_path silently resume from this one.
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        return self.model, num_step

    # =========================================================================
    # AC-PDD v2  -  with budget floor                    --mode acpdd_ratio_floor
    # -------------------------------------------------------------------------
    # Identical to ratio_dbpd above except that `floor_fraction` is threaded
    # into ratio_dbpd_select, so target_budget can never fall below
    # floor_fraction * batch_size, plus three extra diagnostic columns
    # (floor_triggered_batches / total_batches / floor_trigger_rate) that say
    # how often the floor actually bound this epoch. A floor_trigger_rate of 0
    # means the natural hard pool was still above the floor every batch, i.e.
    # this run is behaving exactly like plain ratio_dbpd so far.
    #
    # Held as a near-duplicate of ratio_dbpd on purpose: the un-floored method
    # stays a clean comparison condition, with no shared code path that a later
    # edit could change underneath it. See the method's own docstring.
    # =========================================================================
    def ratio_dbpd_floor(self, start_revision, task, cls_num_list, lambda_min, lambda_max, gamma, floor_fraction):
        """ratio_dbpd with a minimum per-batch selection size, so the
        training set can't collapse to a near-empty residual late in a long
        run the way plain ratio_dbpd/DBPD do (see the epoch-count analysis
        that motivated this: a 200-epoch run's selection pool shrinks toward
        a few hundred out of 50,000 samples by epoch ~100 and stays there,
        so the back half of the run spends its epochs re-memorizing that
        tiny residual instead of getting genuinely more effective training).

        Kept as its own method rather than changing ratio_dbpd in place, so
        the un-floored version stays available as a clean comparison
        condition - identical to ratio_dbpd in every other respect,
        including the learning-rate schedule (StepLR, gamma=0.98,
        unchanged, deliberately - the LR schedule is a separate concern from
        the selection-pool floor and isn't touched here).

        The floor itself lives in ratio_dbpd_select's `floor_fraction`
        parameter: once a batch's hard-sample count alone would put
        ratio_dbpd's usual budget (target_budget = hard_available) below
        `floor_fraction * batch_size`, the budget is raised to that floor
        and the shortfall is filled by the same lambda_t-weighted
        hard/easy split ratio_dbpd already uses - with the hard pool
        exhausted, that split draws the extra slots uniformly at random
        (torch.randperm) from the batch's easy pool. So the floor doesn't
        introduce a second selection mechanism; it just stops the existing
        one from being allowed to shrink the budget below a minimum, and
        fills that minimum the same way SMRD would (a random draw from
        whatever's left), not by repeating the same fixed hard residual.
        """

        save_path = self.save_path
        self.model.to(self.device)
        if task=='classification':
            criterion = nn.CrossEntropyLoss()
        elif 'longtail':
            train_sampler = None
            idx = self.epochs // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(self.device)
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(self.device)
        # LR schedule intentionally identical to ratio_dbpd/train_with_revision -
        # not part of what this method changes.
        optimizer = optim.AdamW(self.model.parameters(), lr=3e-4)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
        epoch_losses = []
        epoch_accuracies = []
        epoch_test_accuracies = []
        epoch_test_losses = []
        time_per_epoch = []
        survival_log = defaultdict(list)
        label_log = defaultdict(int)
        start_time = time.time()
        num_step = 0
        samples_used_per_epoch = []

        ckpt_path = f"{save_path}_checkpoint.pt"
        start_epoch = 0
        pending_recorder_state = None

        hard_count_epoch0 = None
        hard_count_prev_epoch = None

        # Same per-epoch diagnostics CSV pattern as ratio_dbpd, kept as a
        # separate file/name so a floored and un-floored run can write to
        # sibling directories without clobbering each other's stats.
        ratio_dbpd_stats_path = f"{save_path}_ratio_dbpd_floor_epoch_stats.csv"
        ratio_dbpd_stat_rows = pd.read_csv(ratio_dbpd_stats_path).to_dict("records") if os.path.exists(ratio_dbpd_stats_path) else []

        checkpoint = self._load_checkpoint(ckpt_path)
        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            run_id = checkpoint["run_id"]
            start_epoch = checkpoint["epoch"] + 1
            epoch_losses = checkpoint["epoch_losses"]
            epoch_accuracies = checkpoint["epoch_accuracies"]
            epoch_test_accuracies = checkpoint["epoch_test_accuracies"]
            epoch_test_losses = checkpoint["epoch_test_losses"]
            time_per_epoch = checkpoint["time_per_epoch"]
            samples_used_per_epoch = checkpoint["samples_used_per_epoch"]
            num_step = checkpoint["num_step"]
            survival_log = defaultdict(list, checkpoint["survival_log"])
            label_log = defaultdict(int, checkpoint["label_log"])
            pending_recorder_state = checkpoint["recorder_state"]
            hard_count_epoch0 = checkpoint.get("hard_count_epoch0")
            hard_count_prev_epoch = checkpoint.get("hard_count_prev_epoch")
            print(f"[ratio_dbpd_floor] Resumed from checkpoint: epoch {start_epoch}/{self.epochs} (run_id={run_id})")
        else:
            run_id = self.run_id if self.run_id else make_run_id(self.model_name)

        indexed_train_loader = DataLoader(IndexedDataset(self.train_loader.dataset), batch_size=self.train_loader.batch_size, shuffle=True)
        recorder = MetricsRecorder(
            self.model_name, save_path, run_id=run_id, seed=self.seed, dataset=self.dataset, architecture=self.model_name,
            optimizer_name="AdamW", learning_rate=3e-4, total_epochs=self.epochs, batch_size=self.train_loader.batch_size,
            threshold=self.threshold, start_revision=start_revision, mode="ratio_dbpd_floor",
            floor_fraction=floor_fraction,
        )
        if pending_recorder_state is not None:
            recorder.import_state(pending_recorder_state)
        epoch0_sample_ids, epoch0_labels = [], []

        for epoch in range(start_epoch, self.epochs):
            samples_used = 0
            current_lr = optimizer.param_groups[0]["lr"]
            # hard_fraction is kept only for logging/checkpoint compatibility
            # with ratio_dbpd - it doesn't drive selection here either (see
            # ratio_dbpd_select).
            hard_fraction = 1.0 if hard_count_epoch0 is None else (hard_count_prev_epoch / hard_count_epoch0)
            hard_samples_present = 0
            # Counted directly from what actually got backpropped (see the fix
            # below), not derived by subtracting from samples_used - that
            # subtraction went negative because it summed easy over a
            # different set of batches than samples_used did.
            hard_used_count_this_epoch = 0
            easy_used_count_this_epoch = 0
            lambda_t = get_lambda_t(epoch, self.epochs, lambda_min, lambda_max, gamma, start_revision)

            # Same diagnostics as ratio_dbpd, plus floor_triggered_batches -
            # how many batches this epoch actually needed the floor (0 means
            # the natural hard pool was still >= floor_fraction * batch_size
            # every batch, i.e. the floor hasn't started biting yet).
            ratio_dbpd_raw_hard_available = 0
            ratio_dbpd_raw_easy_available = 0
            ratio_dbpd_target_hard_count = 0
            ratio_dbpd_target_easy_count = 0
            ratio_dbpd_selected_hard_count = 0
            ratio_dbpd_selected_easy_count = 0
            ratio_dbpd_target_budget = 0
            ratio_dbpd_actual_selected_count = 0
            floor_triggered_batches = 0
            total_batches = 0

            if epoch < start_revision :
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total_correct = 0
                total_samples = 0
                total = 0
                print(f"Epoch [{epoch+1/self.epochs}]")
                progress_bar = tqdm(enumerate(indexed_train_loader), total=len(indexed_train_loader), desc="Training")

                for batch_idx, (inputs, labels, sample_ids) in progress_bar:
                    batch_start_idx = batch_idx * self.train_loader.batch_size
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        prob = torch.softmax(outputs, dim=1)
                        correct_class = prob[torch.arange(labels.size(0)), labels]
                        sample_losses = per_sample_loss(outputs, labels)

                        if self.threshold == 0:
                            hard_mask = preds != labels
                        else:
                            hard_mask = correct_class < self.threshold
                        easy_mask = ~hard_mask

                        # Only difference from ratio_dbpd: floor_fraction is
                        # passed through, so target_budget can't fall below
                        # floor_fraction * batch_size (see ratio_dbpd_select).
                        retained_hard_mask, retained_easy_mask, sel_stats = ratio_dbpd_select(
                            hard_mask, easy_mask, lambda_t, floor_fraction=floor_fraction,
                        )

                        hard_samples_present += sel_stats["raw_hard_available"]
                        ratio_dbpd_raw_hard_available += sel_stats["raw_hard_available"]
                        ratio_dbpd_raw_easy_available += sel_stats["raw_easy_available"]
                        ratio_dbpd_target_hard_count += sel_stats["target_hard_count"]
                        ratio_dbpd_target_easy_count += sel_stats["target_easy_count"]
                        ratio_dbpd_selected_hard_count += sel_stats["selected_hard_count"]
                        ratio_dbpd_selected_easy_count += sel_stats["selected_easy_count"]
                        ratio_dbpd_target_budget += sel_stats["dbpd_target_budget"]
                        ratio_dbpd_actual_selected_count += sel_stats["actual_selected_count"]
                        floor_triggered_batches += int(sel_stats["floor_triggered"])
                        total_batches += 1

                        mask = retained_hard_mask | retained_easy_mask

                    recorder.log_batch(epoch, sample_ids.tolist(), labels, preds, outputs, sample_losses, correct_class, mask)
                    if epoch == 0:
                        epoch0_sample_ids.extend(sample_ids.tolist())
                        epoch0_labels.extend(labels.detach().cpu().tolist())

                    if mask.sum().item() < 2:
                        continue

                    # Counted here, not above: a batch that fails the >=2 gate
                    # contributes nothing to samples_used, so it must
                    # contribute nothing here either, or hard_used + easy_used
                    # stops summing to samples_used.
                    hard_used_count_this_epoch += retained_hard_mask.sum().item()
                    easy_used_count_this_epoch += retained_easy_mask.sum().item()

                    inputs_misclassified = inputs[mask]
                    labels_misclassified = labels[mask]

                    used_labels = labels_misclassified
                    for label in used_labels.tolist():
                        label_log[int(label)] += 1

                    misclassified_in_batch = torch.nonzero(mask, as_tuple=False).squeeze(1)
                    absolute_indices = (misclassified_in_batch + batch_start_idx).tolist()
                    survival_log[epoch].extend(absolute_indices)

                    optimizer.zero_grad()

                    outputs_misclassified = self.model(inputs_misclassified)
                    loss = criterion(outputs_misclassified, labels_misclassified)
                    num_step+=len(outputs_misclassified)
                    samples_used+=len(outputs_misclassified)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.size(0)
                    progress_bar.set_postfix({"Loss": loss.item()})

                if hard_count_epoch0 is None:
                    hard_count_epoch0 = hard_samples_present
                hard_count_prev_epoch = hard_samples_present

                budget_fill_rate = (ratio_dbpd_actual_selected_count / ratio_dbpd_target_budget) if ratio_dbpd_target_budget > 0 else 1.0
                realised_hard_proportion = (ratio_dbpd_selected_hard_count / ratio_dbpd_actual_selected_count) if ratio_dbpd_actual_selected_count > 0 else 0.0
                realised_easy_proportion = (ratio_dbpd_selected_easy_count / ratio_dbpd_actual_selected_count) if ratio_dbpd_actual_selected_count > 0 else 0.0
                floor_trigger_rate = (floor_triggered_batches / total_batches) if total_batches > 0 else 0.0
                ratio_dbpd_stat_rows.append({
                    "epoch": epoch, "phase": "selection", "lambda_t": lambda_t,
                    "floor_fraction": floor_fraction,
                    "raw_hard_available": ratio_dbpd_raw_hard_available,
                    "raw_easy_available": ratio_dbpd_raw_easy_available,
                    "target_hard_count": ratio_dbpd_target_hard_count,
                    "target_easy_count": ratio_dbpd_target_easy_count,
                    "selected_hard_count": ratio_dbpd_selected_hard_count,
                    "selected_easy_count": ratio_dbpd_selected_easy_count,
                    "dbpd_target_budget": ratio_dbpd_target_budget,
                    "actual_selected_count": ratio_dbpd_actual_selected_count,
                    "budget_fill_rate": budget_fill_rate,
                    "realised_hard_proportion": realised_hard_proportion,
                    "realised_easy_proportion": realised_easy_proportion,
                    "floor_triggered_batches": floor_triggered_batches,
                    "total_batches": total_batches,
                    "floor_trigger_rate": floor_trigger_rate,
                    # Diagnostic-only, unused by selection - see hard_fraction's comment above.
                    "hard_fraction_diagnostic": hard_fraction,
                })
                pd.DataFrame(ratio_dbpd_stat_rows).to_csv(ratio_dbpd_stats_path, index=False)
                print(f"[ratio_dbpd_floor] Epoch {epoch + 1}/{self.epochs}: "
                      f"hard={ratio_dbpd_selected_hard_count}, easy={ratio_dbpd_selected_easy_count}, "
                      f"total={ratio_dbpd_actual_selected_count} (dbpd_target_budget={ratio_dbpd_target_budget}, "
                      f"floor hit {floor_triggered_batches}/{total_batches} batches)")

                if epoch == 0:
                    recorder.log_identifiers(epoch0_sample_ids, epoch0_labels)

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = total_correct/total_samples if total_samples > 0 else 0
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time-epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                correct = 0
                total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)

                accuracy = correct / total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step()
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

                recorder.log_epoch(
                    epoch=epoch,
                    learning_rate=current_lr,
                    threshold=self.threshold,
                    train_loss=epoch_loss,
                    train_accuracy=epoch_accuracy,
                    val_loss=val_loss,
                    val_accuracy=accuracy,
                    samples_used=samples_used,
                    samples_total=total_samples,
                    time_seconds=time_per_epoch[-1],
                    phase="selection",
                    hard_fraction=hard_fraction,
                    lambda_t=lambda_t,
                    # hard_used_count_this_epoch and easy_used_count_this_epoch
                    # are each counted directly from the same >=2-gated
                    # batches samples_used comes from (see above), so the two
                    # sum to samples_used exactly - do not replace either with
                    # samples_used minus the other: that subtraction
                    # previously summed easy over every batch while
                    # samples_used only summed over ungated ones, which went
                    # negative once most batches were skipped.
                    hard_available_count=hard_samples_present,
                    easy_available_count=total_samples - hard_samples_present,
                    hard_used_count=hard_used_count_this_epoch,
                    easy_used_count=easy_used_count_this_epoch,
                )

            else:
                self.model.train()
                epoch_start_time = time.time()
                running_loss = 0.0
                correct = 0
                total = 0
                hard_count_this_epoch = 0

                print(f"Epoch [{epoch+1/self.epochs}]")
                progress_bar = tqdm(enumerate(indexed_train_loader), total=len(indexed_train_loader), desc="Training")


                for batch_idx, (inputs, labels, sample_ids) in progress_bar:
                    batch_start_idx = batch_idx * self.train_loader.batch_size
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    absolute_indices = list(range(batch_start_idx, batch_start_idx + inputs.size(0)))
                    survival_log[epoch].extend(absolute_indices)
                    used_labels = labels
                    for label in used_labels.tolist():
                        label_log[int(label)] += 1
                    num_step+=len(outputs)
                    samples_used+=len(outputs)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    with torch.no_grad():
                        outputs = self.model(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        prob = torch.softmax(outputs, dim=1)
                        correct_class = prob[torch.arange(labels.size(0)), labels]
                        sample_losses = per_sample_loss(outputs, labels)
                        kept_mask = torch.ones_like(labels, dtype=torch.bool)  # revision phase: all samples kept
                        if self.threshold == 0:
                            hard_count_this_epoch += (preds != labels).sum().item()
                        else:
                            hard_count_this_epoch += (correct_class < self.threshold).sum().item()
                        recorder.log_batch(epoch, sample_ids.tolist(), labels, preds, outputs, sample_losses, correct_class, kept_mask)
                        if epoch == 0:
                            epoch0_sample_ids.extend(sample_ids.tolist())
                            epoch0_labels.extend(labels.detach().cpu().tolist())
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                if epoch == 0:
                    recorder.log_identifiers(epoch0_sample_ids, epoch0_labels)

                epoch_loss = running_loss / len(self.train_loader)
                epoch_accuracy = correct / total
                epoch_losses.append(epoch_loss)
                epoch_accuracies.append(epoch_accuracy)

                epoch_end_time = time.time()
                time_per_epoch.append(epoch_end_time - epoch_start_time)

                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

                self.model.eval()
                test_correct = 0
                test_total = 0
                test_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.test_loader, desc="Evaluating"):
                        inputs = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        outputs = self.model(inputs)

                        batch_loss = criterion(outputs, labels)
                        test_loss+=batch_loss.item()

                        predictions = torch.argmax(outputs, dim=-1)
                        test_correct += (predictions == labels).sum().item()
                        test_total += labels.size(0)

                accuracy = test_correct / test_total
                val_loss = test_loss / len(self.test_loader)
                print(f"Epoch {epoch + 1}/{self.epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
                scheduler.step()
                epoch_test_accuracies.append(accuracy)
                epoch_test_losses.append(val_loss)

                # No selection runs during revision (full dataset,
                # unweighted, unchanged) - a NaN-filled row keeps the CSV
                # epoch-indexed without implying selection stats that were
                # never computed this epoch.
                ratio_dbpd_stat_rows.append({
                    "epoch": epoch, "phase": "revision", "lambda_t": lambda_t,
                    "floor_fraction": floor_fraction,
                    "raw_hard_available": float("nan"), "raw_easy_available": float("nan"),
                    "target_hard_count": float("nan"), "target_easy_count": float("nan"),
                    "selected_hard_count": float("nan"), "selected_easy_count": float("nan"),
                    "dbpd_target_budget": float("nan"), "actual_selected_count": float("nan"),
                    "budget_fill_rate": float("nan"), "realised_hard_proportion": float("nan"),
                    "realised_easy_proportion": float("nan"),
                    "floor_triggered_batches": float("nan"), "total_batches": float("nan"),
                    "floor_trigger_rate": float("nan"), "hard_fraction_diagnostic": hard_fraction,
                })
                pd.DataFrame(ratio_dbpd_stat_rows).to_csv(ratio_dbpd_stats_path, index=False)
                print(f"[ratio_dbpd_floor] Epoch {epoch + 1}/{self.epochs} (revision): "
                      f"hard={hard_count_this_epoch}, easy={total - hard_count_this_epoch}, "
                      f"total={total} (full dataset, no selection)")

                recorder.log_epoch(
                    epoch=epoch,
                    learning_rate=current_lr,
                    threshold=self.threshold,
                    train_loss=epoch_loss,
                    train_accuracy=epoch_accuracy,
                    val_loss=val_loss,
                    val_accuracy=accuracy,
                    samples_used=samples_used,
                    samples_total=total,
                    time_seconds=time_per_epoch[-1],
                    phase="revision",
                    hard_fraction=hard_fraction,
                    lambda_t=lambda_t,
                    hard_available_count=hard_count_this_epoch,
                    easy_available_count=total - hard_count_this_epoch,
                    hard_used_count=hard_count_this_epoch,
                    easy_used_count=total - hard_count_this_epoch,
                )

            samples_used_per_epoch.append(samples_used)

            recorder.finalize_epoch_samples()
            self._save_checkpoint(
                ckpt_path, epoch, run_id, optimizer, scheduler,
                epoch_losses=epoch_losses, epoch_accuracies=epoch_accuracies,
                epoch_test_accuracies=epoch_test_accuracies, epoch_test_losses=epoch_test_losses,
                time_per_epoch=time_per_epoch, samples_used_per_epoch=samples_used_per_epoch,
                num_step=num_step, survival_log=dict(survival_log), label_log=dict(label_log),
                recorder_state=recorder.export_state(),
                hard_count_epoch0=hard_count_epoch0, hard_count_prev_epoch=hard_count_prev_epoch,
            )

        self.model.eval()
        final_ids, final_labels, final_preds, final_logits, final_losses, final_probs = [], [], [], [], [], []
        with torch.no_grad():
            for inputs, labels, sample_ids in tqdm(indexed_train_loader, desc="Final outcomes pass"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                prob = torch.softmax(outputs, dim=1)
                correct_class = prob[torch.arange(labels.size(0)), labels]
                sample_losses = per_sample_loss(outputs, labels)
                final_ids.extend(sample_ids.tolist())
                final_labels.append(labels)
                final_preds.append(preds)
                final_logits.append(outputs)
                final_losses.append(sample_losses)
                final_probs.append(correct_class)
        recorder.log_outcomes(
            final_ids, torch.cat(final_labels), torch.cat(final_preds), torch.cat(final_logits),
            torch.cat(final_losses), torch.cat(final_probs),
        )

        recorder.merge_epoch_samples()
        recorder.close()
        end_time = time.time()
        log_memory(start_time, end_time)
        print(num_step)

        total_wall_time = end_time - start_time
        print(f"\n Total Wall Time for {self.epochs} epochs: {total_wall_time:.2f} seconds "
            f"({total_wall_time / 60:.2f} minutes)")

        plot_accuracy_time_multi(
            model_name=self.model_name,
            accuracy=epoch_accuracies,
            time_per_epoch=time_per_epoch,
            save_path=save_path,
            data_file=save_path,
        )
        plot_accuracy_time_multi_test(
            model_name=self.model_name,
            accuracy=epoch_test_accuracies,
            time_per_epoch=time_per_epoch,
            samples_per_epoch=samples_used_per_epoch,
            threshold=self.threshold,
            save_path=save_path,
            data_file=save_path,
        )

        survival_log_path = os.path.join(os.path.dirname(save_path), "survival_log_eff.json")
        with open(survival_log_path, "w") as f:
            json.dump(dict(survival_log), f, indent=2)
        print(f"Survival log saved to {survival_log_path}")

        label_log_path = os.path.join(os.path.dirname(save_path), "label_log_eff.json")
        with open(label_log_path, "w") as f:
            json.dump(dict(label_log), f, indent=2)
        print(f"Survival log saved to {label_log_path}")

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

        return self.model, num_step
