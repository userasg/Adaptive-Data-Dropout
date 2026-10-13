"""
main.py - single entry point for all training runs.
===================================================

Repository layout (reorganised so each variant maps 1:1 onto a paper section):

    pdd/          Progressive Data Dropout      (NeurIPS 2025)  -> paper/PDD.pdf
    add/          Adaptive Data Dropout         (ADD)           -> paper/ADD.pdf
    ac_pdd/       AC-PDD, adaptive-composition PDD (in NEITHER paper)
    experiments/  exploratory variants in NEITHER paper

Every training function returns (model, num_step), where num_step is the number
of samples that went through BACKPROPAGATION over the whole run. Effective
Epochs = num_step / dataset_size (PDD S3.5 / ADD S3.5).

Run `python main.py --help` for the full mode table.
"""

import argparse
import random

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from model_zoo import ModelZoo
from data import (
    load_cifar100, load_mnist, load_imagenet, load_cityscapes,
    load_cifar10, load_medmnist3D
)
from baseline import train_baseline
from longtail_train import train_baseline_longtail, train_with_revision_longtail

# --- PDD: Progressive Data Dropout (paper/PDD.pdf) ---------------------------
from pdd.pdd_training_variants import TrainRevision

# --- ADD: Adaptive Data Dropout (paper/ADD.pdf) ------------------------------
from add.adaptive_alpha import SimAnnealDropoutSimple, SAConfig as AdaptiveAlphaConfig
from add.adaptive_keep_fraction import SimAnnealDropout, SAConfig as AdaptiveTConfig

# --- experiments: not in either paper ----------------------------------------
from experiments.echo_srd import EchoSRD, EchoSRDConfig
from experiments.genetic_dropout import GaDropout
from experiments.simple_switcher import SimpleSwitcher
import experiments.genetic_revision as GA


# =============================================================================
# Mode registry
# =============================================================================
# PDD modes, all served by the single TrainRevision class in
# pdd/pdd_training_variants.py.  call style:
#   "none"         -> method()                              returns model only
#   "rev"          -> method(start_revision)
#   "rev_task"     -> method(start_revision, task)
#   "rev_datasize" -> method(start_revision, data_size)
PDD_MODES = {
    # mode name               (method on TrainRevision,        call style)
    "pdd_dbpd":               ("train_with_revision",           "rev_task"),
    "pdd_srd":                ("train_with_percentage",         "rev"),
    "pdd_smrd":               ("train_with_random",             "rev_task"),
    "pdd_dbpd_3d":            ("train_with_revision_3d",        "rev_task"),
    "pdd_approx_inv_lin":     ("train_with_inverse_linear",     "rev_datasize"),
    "pdd_approx_log":         ("train_with_log",                "rev_datasize"),
    "pdd_approx_power_law":   ("train_with_power_law",          "rev_datasize"),
    "pdd_approx_exponential": ("train_with_exponential",        "rev_datasize"),
    "pdd_approx_sigmoid":     ("train_with_sigmoid_complement", "rev_datasize"),
    "pdd_selective_gradient": ("train_selective",               "none"),
    "pdd_selective_epoch":    ("train_selective_epoch",         "none"),
}

ADD_MODES = ("add_adaptive_alpha", "add_adaptive_t")

# AC-PDD: DBPD's budget with a lambda_t-scheduled hard/easy composition.
# Served by ac_pdd/ratio_dbpd.py's own RatioDBPD class, NOT by TrainRevision -
# it carries MetricsRecorder instrumentation and resume support that the PDD
# class does not have, so sharing the class would have changed all pdd_* modes.
ACPDD_MODES = ("acpdd_ratio", "acpdd_ratio_floor")

EXP_MODES = ("exp_echo_srd", "exp_genetic", "exp_genetic_legacy", "exp_switcher")

# Old mode names kept working so existing shell scripts / history still run.
LEGACY_ALIASES = {
    "train_with_revision":           "pdd_dbpd",
    "train_with_percentage":         "pdd_srd",
    "train_with_samples":            "pdd_srd",
    "train_with_random":             "pdd_smrd",
    "train_with_revision_3d":        "pdd_dbpd_3d",
    "train_with_inv_lin":            "pdd_approx_inv_lin",
    "train_with_log":                "pdd_approx_log",
    "train_with_power_law":          "pdd_approx_power_law",
    "train_with_exponential":        "pdd_approx_exponential",
    "train_with_sigmoid_complement": "pdd_approx_sigmoid",
    "selective_gradient":            "pdd_selective_gradient",
    "selective_epoch":               "pdd_selective_epoch",
    "saRevisionSimple":              "add_adaptive_alpha",
    "train_with_sa":                 "add_adaptive_t",
    # AC_PDD repo's own mode names, so its commands run here unchanged
    "ratio_dbpd":                    "acpdd_ratio",
    "ratio_dbpd_floor":              "acpdd_ratio_floor",
    "EchoSRD_with_revision":         "exp_echo_srd",
    "gaRevisionFinal":               "exp_genetic",
    "train_with_genetic":            "exp_genetic_legacy",
    "train_with_switching":          "exp_switcher",
}

ALL_MODES = (["baseline"] + list(PDD_MODES) + list(ADD_MODES)
             + list(ACPDD_MODES) + list(EXP_MODES))

MODE_HELP = """\
MODES  (grouped by paper; old mode names still work as aliases)

  baseline                  standard training, no dropout

  PDD - Progressive Data Dropout                              paper/PDD.pdf
  pdd_dbpd                  V1  DBPD   difficulty-based             (was train_with_revision)
  pdd_srd                   V2  SRD    scalar random, gamma=0.95    (was train_with_percentage)
  pdd_smrd                  V3  SMRD   schedule-matched random      (was train_with_random)
  pdd_approx_power_law      App.C (1)  power-law decay
  pdd_approx_exponential    App.C (2)  exponential decay
  pdd_approx_log            App.C (3)  logarithmic decay
  pdd_approx_inv_lin        App.C (4)  inverse linear decay         (was train_with_inv_lin)
  pdd_approx_sigmoid        App.C (5)  sigmoid-complement decay
  pdd_dbpd_3d               DBPD for 3D/video inputs
  pdd_selective_gradient    prototype, no revision phase  [known bug, see file]
  pdd_selective_epoch       prototype, gradual reintroduction

  ADD - Adaptive Data Dropout                                 paper/ADD.pdf
  add_adaptive_alpha        S3.2  Adaptive-alpha               (was saRevisionSimple)
  add_adaptive_t            S3.3  Adaptive-T keep-fraction     (was train_with_sa)

  AC-PDD - adaptive-composition PDD, in NEITHER paper        ac_pdd/
  acpdd_ratio               ratio-DBPD: DBPD budget, lambda_t  (was ratio_dbpd)
                            scheduled hard/easy composition
  acpdd_ratio_floor         ratio-DBPD + minimum budget        (was ratio_dbpd_floor)
                            (--floor_fraction)
    --lambda_min F          composition weight at epoch 0   (0 = hard-only, == DBPD)
    --lambda_max F          weight at end of selection      (1 = natural ratio, == SMRD)
    --gamma F               lambda_t ramp exponent, progress**gamma. NOT the
                            StepLR decay factor, which is fixed at 0.98.
    --floor_fraction F      acpdd_ratio_floor only: minimum per-batch budget as
                            a fraction of batch_size.
  NOTE: these two modes write Parquet via ac_pdd/learning_cycles.py and need
        pyarrow installed. No other mode does.

  experiments - in NEITHER paper
  exp_echo_srd              oscillating SRD, no feedback      (was EchoSRD_with_revision)
  exp_genetic               GA over per-sample scores         (was gaRevisionFinal)
  exp_genetic_legacy        older GA scheduler                (was train_with_genetic)
  exp_switcher              switches strategy mid-run         (was train_with_switching)

REVISION PHASE
  --final_revision y   (default) the LAST epoch trains on the full dataset,
                       i.e. the paper's revision pass D_T = D_0. Applies to
                       every pdd_*, add_* and exp_echo_srd mode.
  --final_revision n   no revision phase; dropout runs for every epoch.
  --start_revision N   ADVANCED, pdd_* only: start the full-data phase at epoch
                       N for a revision tail longer than one epoch (e.g. PDD
                       Table 15). Overrides --final_revision when N > 0.
"""


# =============================================================================
# Helpers
# =============================================================================
def make_train_val_loaders(train_loader, val_frac=0.10, seed=42):
    """Carve a validation split out of TRAIN (only exp_genetic_legacy needs this)."""
    ds = train_loader.dataset
    N = len(ds)
    idx = list(range(N))
    rng = random.Random(seed); rng.shuffle(idx)
    v = int(N * val_frac)
    val_idx, train_idx = idx[:v], idx[v:]

    kw = dict(
        batch_size=getattr(train_loader, "batch_size", 128),
        num_workers=getattr(train_loader, "num_workers", 0),
        pin_memory=getattr(train_loader, "pin_memory", False),
        drop_last=getattr(train_loader, "drop_last", False),
        collate_fn=getattr(train_loader, "collate_fn", None),
    )
    kw = {k: v for k, v in kw.items() if v is not None}

    train_loader_split = DataLoader(ds, sampler=SubsetRandomSampler(train_idx), shuffle=False, **kw)
    val_loader = DataLoader(ds, sampler=SubsetRandomSampler(val_idx), shuffle=False, **kw)
    return train_loader_split, val_loader


def pick_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: cuda")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps")
        print("Apple Silicon GPU (MPS) enabled.")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")
        print("Warning: No GPU found. Using CPU.")
    return device


def load_dataset(args):
    """Returns (train_loader, test_loader, num_classes, data_size, cls_num_list)."""
    cls_num_list = None

    if args.dataset == "mnist":
        train_loader, test_loader = load_mnist()
        return train_loader, test_loader, 10, len(train_loader.dataset), cls_num_list

    if args.dataset == "cifar":
        if args.batch_size:
            train_loader, test_loader, cls_num_list, data_size = load_cifar100(args.long_tail, args.batch_size)
        else:
            train_loader, test_loader, cls_num_list, data_size = load_cifar100(args.long_tail)
        return train_loader, test_loader, 100, data_size, cls_num_list

    if args.dataset == "cifar10":
        # guard against --batch_size being omitted (load_cifar10 defaults to 128)
        if args.batch_size:
            train_loader, test_loader, cls_num_list, data_size = load_cifar10(args.long_tail, args.batch_size)
        else:
            train_loader, test_loader, cls_num_list, data_size = load_cifar10(args.long_tail)
        return train_loader, test_loader, 10, data_size, cls_num_list

    if args.dataset == "imagenet":
        train_loader, test_loader, data_size = load_imagenet(args.batch_size)
        return train_loader, test_loader, 1000, data_size, cls_num_list

    if args.dataset == "cityscapes":
        train_loader, test_loader = load_cityscapes()
        return train_loader, test_loader, 19, len(train_loader.dataset), cls_num_list

    if args.dataset == "organ_medmnist3d":
        train_loader, test_loader, data_size = load_medmnist3D(args.batch_size)
        return train_loader, test_loader, 11, data_size, cls_num_list

    raise ValueError(f"Unknown dataset: {args.dataset}")


def build_model(args, num_classes, parser):
    # --model value -> ModelZoo method NAME. Looked up lazily with getattr so a
    # model_zoo method that is missing only breaks the run that asks for it
    # (note: ModelZoo currently has no `segformer`, so segformer_b2 is unavailable).
    builders = {
        "resnet18":        "resnet18",
        "resnet34":        "resnet34",
        "resnet50":        "resnet50",
        "resnet101":       "resnet101",
        "resnet_3d":       "resnet18_3d",
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b4": "efficientnet_b4",
        "efficientnet_b7": "efficientnet_b7",
        "mobilenet_v2":    "mobilenet_v2",
        "mobilenet_v3":    "mobilenet_v3",
        "vit_b_16":        "vit_b_16",
        "efficientformer": "efficientformer",
        "segformer_b2":    "segformer",
    }
    mz = ModelZoo(num_classes, bool(args.pretrained))

    if args.model == "mae_vit_b_16":
        if not args.mae_checkpoint:
            parser.error("--mae_checkpoint is required when using --model mae_vit_b_16")
        return mz.mae_vit_b_16(checkpoint_path=args.mae_checkpoint)

    if args.model not in builders:
        raise ValueError(f"Unknown model: {args.model}")
    method = getattr(mz, builders[args.model], None)
    if method is None:
        raise ValueError(
            f"--model {args.model} maps to ModelZoo.{builders[args.model]}(), "
            f"which does not exist in model_zoo.py.")
    return method()


def str2bool(v):
    """Accept y/n, yes/no, true/false, 1/0 for the --final_revision flag."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("y", "yes", "true", "t", "1", "on"):
        return True
    if s in ("n", "no", "false", "f", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(
        f"expected y/n (or 1/0, true/false), got '{v}'")


def resolve_start_revision(args):
    """
    Turn --final_revision into the epoch index where the full-data phase starts.

    TrainRevision gates on `if epoch < start_revision`:
        start_revision == epochs - 1  -> last epoch trains on ALL data (D_T = D_0)
        start_revision == epochs      -> no revision at all, dropout throughout

    --start_revision stays available as an advanced override for anyone who
    wants a longer revision tail (e.g. PDD Table 15's 25/50/75 revision epochs).
    """
    if args.start_revision is not None and int(args.start_revision) > 0:
        sr = int(args.start_revision)
        print(f"  --start_revision {sr} (explicit override): full-data phase "
              f"covers epochs {sr}..{args.epoch - 1}")
        if not args.final_revision:
            print("  WARNING: --final_revision n is ignored because "
                  "--start_revision was given explicitly.")
        return sr

    if args.final_revision:
        sr = max(args.epoch - 1, 1)
        print(f"  --final_revision y: last epoch trains on the FULL dataset "
              f"(start_revision={sr} of {args.epoch})")
        return sr

    print("  --final_revision n: no revision phase, dropout for every epoch")
    return args.epoch


def run_pdd_mode(mode, args, model, train_loader, test_loader, device, data_size):
    """Dispatch any of the PDD variants through the shared TrainRevision class."""
    method_name, style = PDD_MODES[mode]
    tr = TrainRevision(args.model, model, train_loader, test_loader, device,
                       args.epoch, args.save_path, args.threshold)
    method = getattr(tr, method_name)
    print(f"[PDD] mode={mode} -> TrainRevision.{method_name}()")

    if style == "none":
        # these two prototypes return the model only, with no step count
        return method(), None

    start_revision = resolve_start_revision(args)

    if style == "rev":
        return method(start_revision)
    if style == "rev_task":
        return method(start_revision, args.task)
    if style == "rev_datasize":
        # the Appendix C schedules ignore start_revision (no revision phase)
        return method(start_revision, data_size)
    raise ValueError(f"Unknown call style: {style}")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Training entry point for PDD / ADD / experimental dropout variants.",
        epilog=MODE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=ALL_MODES + list(LEGACY_ALIASES),
                        metavar="MODE", help="see the mode table below")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--task", type=str, default="classification")
    parser.add_argument("--model", type=str, required=True, choices=[
        "resnet18", "resnet_3d", "resnet34", "resnet50", "resnet101",
        "efficientnet_b0", "efficientnet_b7", "efficientnet_b4",
        "mobilenet_v2", "mobilenet_v3",
        "vit_b_16", "mae_vit_b_16",
        "efficientformer", "segformer_b2",
    ])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--mae_checkpoint", type=str, default=None)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--threshold", type=float,
                        help="PDD confidence threshold tau; exp_genetic_legacy maps it to KEEP_FLOOR")
    parser.add_argument("--epoch_threshold", type=int)
    parser.add_argument("--dataset", type=str, required=True,
                        help="cifar | cifar10 | mnist | imagenet | cityscapes | organ_medmnist3d")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--final_revision", type=str2bool, nargs="?", const=True, default=True,
                        metavar="y/n",
                        help="does the LAST epoch train on the full dataset? "
                             "y (default) = the paper's revision phase, D_T = D_0; "
                             "n = dropout for every epoch, no revision. "
                             "Applies to all pdd_*, add_* and exp_echo_srd modes.")
    parser.add_argument("--start_revision", type=int, default=None,
                        help="ADVANCED override for pdd_* modes: first epoch of the "
                             "full-data phase, for a revision tail longer than one "
                             "epoch. Ignored when <= 0; use --final_revision instead.")
    parser.add_argument("--long_tail", action="store_true")
    parser.add_argument("--ldam", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # exp_echo_srd overrides
    parser.add_argument("--echo_schedule", type=str, default=None,
                        choices=["geometric", "linear", "power_law", "exponential", "log",
                                 "inverse_linear", "sigmoid_complement"])
    parser.add_argument("--echo_alpha", type=float, default=None)
    parser.add_argument("--echo_osc", type=float, default=None)
    parser.add_argument("--echo_revision_epochs", type=int, default=None)
    parser.add_argument("--echo_disable_revision", action="store_true")
    # AC-PDD (acpdd_*) overrides. Defaults match the AC_PDD repo's own
    # main.py so a command copied from there behaves identically here.
    parser.add_argument("--lambda_min", type=float, default=0.0,
                        help="[acpdd_*] composition weight at epoch 0 "
                             "(0.0 = hard-only, identical to DBPD)")
    parser.add_argument("--lambda_max", type=float, default=1.0,
                        help="[acpdd_*] composition weight at the end of the "
                             "selection phase (1.0 = the batch's natural "
                             "hard/easy proportion, identical to SMRD in "
                             "expectation)")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="[acpdd_*] exponent shaping the lambda_t ramp "
                             "(progress**gamma; higher = slower start, faster "
                             "finish). NOT the StepLR learning-rate decay "
                             "factor, which is fixed at 0.98.")
    parser.add_argument("--floor_fraction", type=float, default=0.1,
                        help="[acpdd_ratio_floor] minimum per-batch selection "
                             "size as a fraction of batch_size; once the hard "
                             "pool alone would put the budget below this, the "
                             "shortfall is a random draw from the rest of the "
                             "batch")
    args = parser.parse_args()

    # Resolve legacy mode names to their new equivalents.
    mode = LEGACY_ALIASES.get(args.mode, args.mode)
    if mode != args.mode:
        print(f"Note: --mode {args.mode} is a legacy name; running '{mode}'.")

    device = pick_device()
    train_loader, test_loader, num_classes, data_size, cls_num_list = load_dataset(args)
    model = build_model(args, num_classes, parser).to(device)

    # --- name tagging (drives the keys written into the result JSON) ---------
    if args.pretrained:
        args.model = f"{args.model}_pretrained_{args.threshold}"
    else:
        args.model = f"{args.model}_{args.threshold}"
    if mode == "baseline":
        args.model = f"{args.model}_baseline"

    num_step = None  # set by every mode except baseline / the two prototypes

    # =========================================================================
    # Long-tail + LDAM (only baseline and DBPD are implemented in this branch)
    # =========================================================================
    if args.long_tail and args.ldam:
        if mode == "baseline":
            trained_model = train_baseline_longtail(
                args.model, model, train_loader, test_loader,
                device, args.epoch, args.save_path, cls_num_list)
        elif mode == "pdd_dbpd":
            trained_model = train_with_revision_longtail(
                args.model, model, train_loader, test_loader,
                device, args.epoch, args.save_path,
                args.threshold, resolve_start_revision(args), args.task, cls_num_list)
        else:
            raise ValueError(f"Mode '{mode}' is not implemented for --long_tail --ldam "
                             f"(only 'baseline' and 'pdd_dbpd' are).")

    # =========================================================================
    # Baseline
    # =========================================================================
    elif mode == "baseline":
        print("Training in baseline mode...")
        trained_model = train_baseline(args.model, model, train_loader, test_loader,
                                       device, args.epoch, args.save_path)

    # =========================================================================
    # PDD - Progressive Data Dropout  (pdd/pdd_training_variants.py)
    # =========================================================================
    elif mode in PDD_MODES:
        trained_model, num_step = run_pdd_mode(
            mode, args, model, train_loader, test_loader, device, data_size)

    # =========================================================================
    # ADD - Adaptive Data Dropout  (add/)
    # =========================================================================
    elif mode == "add_adaptive_alpha":
        # ADD S3.2 - Adaptive-alpha: |D_t| = f(t; alpha_t) * N
        print("[ADD] Adaptive-alpha (add/adaptive_alpha.py)")
        tr = SimAnnealDropoutSimple(args.model, model, train_loader, test_loader,
                                    device, args.epoch, args.save_path, args.threshold)
        cfg = AdaptiveAlphaConfig(seed=args.seed)
        cfg.revision_flag = args.final_revision
        print(f'  final-epoch full-data revision: {"on" if cfg.revision_flag else "off"}'
              f' ({cfg.revision_epochs} epoch(s))')
        trained_model, num_step = tr.train_with_sa(cfg)

    elif mode == "add_adaptive_t":
        # ADD S3.3 - Adaptive-T: |D_t| = T_t * N, with reheating on rejection
        print("[ADD] Adaptive-T / keep-fraction (add/adaptive_keep_fraction.py)")
        tr = SimAnnealDropout(args.model, model, train_loader, test_loader,
                              device, args.epoch, args.save_path, args.threshold)
        cfg = AdaptiveTConfig(seed=args.seed)
        cfg.revision_flag = args.final_revision
        print(f'  final-epoch full-data revision: {"on" if cfg.revision_flag else "off"}'
              f' ({cfg.revision_epochs} epoch(s))')
        trained_model, num_step = tr.train_with_sa(cfg)

    # =========================================================================
    # AC-PDD - adaptive-composition PDD  (ac_pdd/)
    # In NEITHER paper. Uses its own RatioDBPD class, so pdd/ is untouched.
    # =========================================================================
    elif mode in ACPDD_MODES:
        # Imported HERE rather than at module scope on purpose: ac_pdd pulls in
        # learning_cycles, which imports pyarrow at module level. A top-level
        # import would make a missing/broken pyarrow break every mode in this
        # file, including baseline. Keeping it lazy confines that dependency to
        # the two acpdd_* modes.
        from ac_pdd.learning_cycles import set_seed, make_run_id
        from ac_pdd.ratio_dbpd import RatioDBPD

        # AC-PDD's upstream main.py seeds globally; done only in this branch so
        # no other mode's RNG stream shifts.
        set_seed(args.seed)

        start_revision = resolve_start_revision(args)
        tr = RatioDBPD(args.model, model, train_loader, test_loader, device,
                       args.epoch, args.save_path, args.threshold,
                       run_id=make_run_id(args.model), seed=args.seed,
                       dataset=args.dataset)

        if mode == "acpdd_ratio":
            print("[AC-PDD] ratio-DBPD (ac_pdd/ratio_dbpd.py)")
            print(f"  lambda_min={args.lambda_min}, lambda_max={args.lambda_max}, "
                  f"gamma={args.gamma}, revision starts at epoch {start_revision}")
            trained_model, num_step = tr.ratio_dbpd(
                start_revision, args.task, cls_num_list,
                args.lambda_min, args.lambda_max, args.gamma)
        else:
            print("[AC-PDD] ratio-DBPD + budget floor (ac_pdd/ratio_dbpd.py)")
            print(f"  lambda_min={args.lambda_min}, lambda_max={args.lambda_max}, "
                  f"gamma={args.gamma}, floor_fraction={args.floor_fraction}, "
                  f"revision starts at epoch {start_revision}")
            trained_model, num_step = tr.ratio_dbpd_floor(
                start_revision, args.task, cls_num_list,
                args.lambda_min, args.lambda_max, args.gamma,
                args.floor_fraction)

    # =========================================================================
    # experiments - not in either paper  (experiments/)
    # =========================================================================
    elif mode == "exp_echo_srd":
        print("[EXP] Echo/Alternating SRD (experiments/echo_srd.py)")
        tr = EchoSRD(args.model, model, train_loader, test_loader,
                     device, args.epoch, args.save_path, args.threshold)
        cfg = EchoSRDConfig(seed=args.seed)
        cfg.revision_flag = args.final_revision
        if args.echo_schedule is not None:
            cfg.schedule = args.echo_schedule
        if args.echo_alpha is not None:
            cfg.alpha = float(args.echo_alpha)
        if args.echo_osc is not None:
            cfg.osc_factor = float(args.echo_osc)
        if args.echo_revision_epochs is not None:
            cfg.revision_epochs = int(args.echo_revision_epochs)
        if args.echo_disable_revision:
            cfg.revision_flag = False
        trained_model, num_step = tr.train_with_echo_srd(cfg)

    elif mode == "exp_genetic":
        print("[EXP] GA per-sample dropout (experiments/genetic_dropout.py)")
        ga = GaDropout(args.model, model, train_loader, test_loader,
                       device, args.epoch, args.save_path, args.threshold, seed=args.seed)
        trained_model, num_step = ga.train_with_genetic()

    elif mode == "exp_genetic_legacy":
        print("[EXP] legacy GA scheduler (experiments/genetic_revision.py)")
        # this one needs a validation split carved out of TRAIN
        train_loader_split, val_loader = make_train_val_loaders(
            train_loader, val_frac=0.10, seed=args.seed)
        if args.threshold is not None:
            GA.KEEP_FLOOR = float(args.threshold)
        ga = GA.GeneticRevision(args.model, model, train_loader_split, val_loader,
                                test_loader, device, args.epoch, args.save_path, seed=args.seed)
        trained_model, num_step = ga.train_with_genetic()

    elif mode == "exp_switcher":
        print("[EXP] dynamic switching scheduler (experiments/simple_switcher.py)")
        switcher = SimpleSwitcher(
            args.model, model, train_loader, test_loader, device, args.epoch,
            args.save_path, args.threshold, data_size,
            start_revision=args.start_revision or 0,
            seed=args.seed)
        trained_model, num_step = switcher.train_with_switching()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # =========================================================================
    # Effective Epochs  (PDD S3.5 / ADD S3.5)
    #   EE = samples through backpropagation / total training samples
    # Baseline and the two prototypes report no step count, so we fall back to
    # the full-data assumption: every sample backprops once per epoch.
    # =========================================================================
    if num_step is None:
        num_step = args.epoch * data_size
        print("Note: this mode does not report a step count; "
              "assuming full-data training for the Effective Epochs figure.")
    print("Number of steps : ", num_step)
    print(f"Effective Epochs: {num_step / data_size:.4f}  (int: {int(num_step / data_size)})")

    torch.save(trained_model, "trained_model.pth")


if __name__ == "__main__":
    main()
