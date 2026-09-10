import argparse
import random
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import resnet18, efficientnet_b0
from model_zoo import ModelZoo
from data import (
    load_cifar100, load_mnist, load_imagenet, load_cityscapes,
    load_cifar10, load_medmnist3D
)
from baseline import train_baseline
from selective_gradient import TrainRevision
from test import test_model
from longtail_train import train_baseline_longtail, train_with_revision_longtail
from SimpleSwitcher import SimpleSwitcher
import GeneticRevision as GA
from SARevision import SimAnnealDropout, SAConfig
from saRevisionSimple import SimAnnealDropoutSimple, SAConfig as SASimpleConfig
from gaDropout import GaDropout
from EchoSRD_with_revision import EchoSRD, EchoSRDConfig

# ---------- tiny helper: carve a validation split out of TRAIN ----------
def make_train_val_loaders(train_loader, val_frac=0.10, seed=42):
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
    # remove Nones for DataLoader kwargs
    kw = {k: v for k, v in kw.items() if v is not None}

    train_loader_split = DataLoader(ds, sampler=SubsetRandomSampler(train_idx), shuffle=False, **kw)
    val_loader = DataLoader(ds, sampler=SubsetRandomSampler(val_idx), shuffle=False, **kw)
    return train_loader_split, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train entry")
    parser.add_argument("--mode", type=str, choices=[
        "baseline", "selective_gradient", "selective_epoch",
        "train_with_revision", "train_with_samples", "train_with_revision_3d",
        "train_with_random", "train_with_inv_lin", "train_with_log",
        "train_with_percentage", "train_with_power_law", "train_with_exponential",
        "train_with_sigmoid_complement", "train_with_switching", "train_with_genetic",
        "train_with_sa", "saRevisionSimple", "gaRevisionFinal",
        "EchoSRD_with_revision"], required=True)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--task", type=str, required=True, default="classification")
    parser.add_argument("--model", type=str, choices=[
        "resnet18", "resnet_3d", "resnet34", "resnet50", "resnet101",
        "efficientnet_b0","efficientnet_b7", "efficientnet_b4",
        "mobilenet_v2", "mobilenet_v3",
        "vit_b_16", "mae_vit_b_16",
        "efficientformer", "segformer_b2"
    ], required=True)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--mae_checkpoint", type=str, default=None)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--threshold", type=float)  # used by many modes; GA can map it to KEEP_FLOOR
    parser.add_argument("--epoch_threshold", type=int)
    parser.add_argument("--dataset", type=str, required=True)  # e.g. cifar, cifar10, mnist, imagenet
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--start_revision", type=int)
    parser.add_argument("--long_tail", action="store_true")
    parser.add_argument("--ldam", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--echo_schedule", type=str, default=None,
                        choices=["geometric", "linear", "power_law", "exponential", "log",
                                 "inverse_linear", "sigmoid_complement"])
    parser.add_argument("--echo_alpha", type=float, default=None)
    parser.add_argument("--echo_osc", type=float, default=None)
    parser.add_argument("--echo_revision_epochs", type=int, default=None)
    parser.add_argument("--echo_disable_revision", action="store_true")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    elif device.type == "mps":
        print("Apple Silicon GPU (MPS) enabled.")
    else:
        print("Warning: No GPU found. Using CPU.")

    # -------------------- load dataset --------------------
    pretrained = bool(args.pretrained)

    if args.dataset == "mnist":
        num_classes = 10
        train_loader, test_loader = load_mnist()
        data_size = len(train_loader.dataset)

    elif args.dataset == "cifar":
        if args.batch_size:
            train_loader, test_loader, cls_num_list, data_size = load_cifar100(args.long_tail, args.batch_size)
        else:
            train_loader, test_loader, cls_num_list, data_size = load_cifar100(args.long_tail)
        num_classes = 100

    elif args.dataset == "cifar10":
        train_loader, test_loader, cls_num_list, data_size = load_cifar10(args.long_tail, args.batch_size)
        num_classes = 10

    elif args.dataset == "imagenet":
        num_classes = 1000
        train_loader, test_loader, data_size = load_imagenet(args.batch_size)

    elif args.dataset == "cityscapes":
        num_classes = 19
        train_loader, test_loader = load_cityscapes()
        data_size = len(train_loader.dataset)

    elif args.dataset == "organ_medmnist3d":
        num_classes = 11
        train_loader, test_loader, data_size = load_medmnist3D(args.batch_size)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # -------------------- create model --------------------
    mz = ModelZoo(num_classes, pretrained)

    if args.model == "resnet18":
        model = mz.resnet18()
    elif args.model == "efficientnet_b0":
        model = mz.efficientnet_b0()
    elif args.model == "mobilenet_v2":
        model = mz.mobilenet_v2()
    elif args.model == "mobilenet_v3":
        model = mz.mobilenet_v3()
    elif args.model == "resnet34":
        model = mz.resnet34()
    elif args.model == "resnet50":
        model = mz.resnet50()
    elif args.model == "resnet101":
        model = mz.resnet101()
    elif args.model == "vit_b_16":
        model = mz.vit_b_16()
    elif args.model == "mae_vit_b_16":
        if not args.mae_checkpoint:
            parser.error("--mae_checkpoint is required when using --model mae_vit_b_16")
        model = mz.mae_vit_b_16(checkpoint_path=args.mae_checkpoint)
    elif args.model == "efficientformer":
        model = mz.efficientformer()
    elif args.model == "efficientnet_b7":
        model = mz.efficientnet_b7()
    elif args.model == "efficientnet_b4":
        model = mz.efficientnet_b4()
    elif args.model == "segformer_b2":
        model = mz.segformer()
    elif args.model == "resnet_3d":
        model = mz.resnet18_3d()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)

    # -------------------- name tagging --------------------
    if args.pretrained:
        args.model = f"{args.model}_pretrained_{args.threshold}"
    else:
        args.model = f"{args.model}_{args.threshold}"

    if args.mode == "baseline":
        args.model = f"{args.model}_baseline"

    # Fallback num_step (for modes that don't return it)
    num_step = args.epoch * data_size

    # -------------------- training modes --------------------
    if args.long_tail and args.ldam:
        if args.mode == "baseline":
            trained_model = train_baseline_longtail(args.model, model, train_loader, test_loader,
                                                    device, args.epoch, args.save_path, cls_num_list)
        elif args.mode == "train_with_revision":
            trained_model = train_with_revision_longtail(args.model, model, train_loader, test_loader,
                                                         device, args.epoch, args.save_path,
                                                         args.threshold, args.start_revision, args.task, cls_num_list)
        else:
            raise ValueError("Selected long-tail+LDAM mode not implemented in this branch.")
    else:
        if args.mode == "baseline":
            print("Training in baseline mode...")
            trained_model = train_baseline(args.model, model, train_loader, test_loader,
                                           device, args.epoch, args.save_path)

        elif args.mode == "selective_gradient":
            tr = TrainRevision(args.model, model, train_loader, test_loader, device,
                               args.epoch, args.save_path, args.threshold)
            print("Training with selective gradient updates...")
            trained_model = tr.train_selective()

        elif args.mode == "selective_epoch":
            tr = TrainRevision(args.model, model, train_loader, test_loader, device,
                               args.epoch, args.save_path, args.threshold)
            print("Reintroducing correct examples and training...")
            trained_model = tr.train_selective_epoch()

        elif args.mode == "train_with_revision":
            tr = TrainRevision(args.model, model, train_loader, test_loader, device,
                               args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = tr.train_with_revision(args.start_revision, args.task)
            print("Number of steps : ", num_step)

        elif args.mode == "train_with_random":
            tr = TrainRevision(args.model, model, train_loader, test_loader, device,
                               args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = tr.train_with_random(args.start_revision, args.task)
            print("Number of steps : ", num_step)

        elif args.mode == "train_with_revision_3d":
            tr = TrainRevision(args.model, model, train_loader, test_loader, device,
                               args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = tr.train_with_revision_3d(args.start_revision, args.task)
            print("Number of steps : ", num_step)

        elif args.mode == "train_with_percentage":
            tr = TrainRevision(args.model, model, train_loader, test_loader, device,
                               args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = tr.train_with_percentage(args.start_revision, args.task)
            print("Number of steps : ", num_step)

        elif args.mode == "train_with_inv_lin":
            tr = TrainRevision(args.model, model, train_loader, test_loader, device,
                               args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = tr.train_with_inverse_linear(args.start_revision, data_size)
            print("Number of steps : ", num_step)

        elif args.mode == "train_with_log":
            tr = TrainRevision(args.model, model, train_loader, test_loader, device,
                               args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = tr.train_with_log(args.start_revision, data_size)
            print("Number of steps : ", num_step)

        elif args.mode == "train_with_power_law":
            tr = TrainRevision(args.model, model, train_loader, test_loader, device,
                               args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = tr.train_with_power_law(args.start_revision, data_size)
            print("Number of steps : ", num_step)

        elif args.mode == "train_with_exponential":
            tr = TrainRevision(args.model, model, train_loader, test_loader, device,
                               args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = tr.train_with_exponential(args.start_revision, data_size)
            print("Number of steps : ", num_step)

        elif args.mode == "train_with_sigmoid_complement":
            tr = TrainRevision(args.model, model, train_loader, test_loader, device,
                               args.epoch, args.save_path, args.threshold)
            print(f"Training {args.mode}, will start revision after {args.start_revision}")
            trained_model, num_step = tr.train_with_sigmoid_complement(args.start_revision, data_size)
            print("Number of steps : ", num_step)

        elif args.mode == "train_with_switching":
            train_switcher = SimpleSwitcher(
                args.model, model, train_loader, test_loader, device, args.epoch,
                args.save_path, args.threshold, data_size,
                start_revision=args.start_revision if args.start_revision else 0,
                seed=args.seed
            )
            print("Training with dynamic switching scheduler…")
            trained_model, num_step = train_switcher.train_with_switching()
            print("Number of steps : ", num_step)

        elif args.mode == "train_with_genetic":
            print("Training with GA-driven dropout schedules…")

            # carve 10% of TRAIN as validation for GA (LR scheduling + monitoring)
            train_loader_split, val_loader = make_train_val_loaders(train_loader, val_frac=0.10, seed=args.seed)

            # allow CLI --threshold to control GA plateau keep%
            if args.threshold is not None:
                GA.KEEP_FLOOR = float(args.threshold)

            ga = GA.GeneticRevision(
                args.model,
                model,
                train_loader_split,   # TRAIN (post-split)
                val_loader,           # VALIDATION (10% of train)
                test_loader,          # TEST (held-out)
                device,
                args.epoch,
                args.save_path,
                seed=args.seed
            )
            trained_model, num_step = ga.train_with_genetic()
            print("Number of steps:", num_step)
        
        elif args.mode == "train_with_sa":
          tr = SimAnnealDropout(
              args.model, model, train_loader, test_loader, device,
              args.epoch, args.save_path, args.threshold
          )
          # Use defaults from SARevision.GLOBAL_SA_PARAMS unless you wire a CLI flag.
          # If you later add --sa_schedule, you can pass schedule=args.sa_schedule.
          cfg = SAConfig(seed=args.seed)  # <-- no hard-coded "power" here
          trained_model, num_step = tr.train_with_sa(cfg)
          print("Number of steps:", num_step)


        elif args.mode == "saRevisionSimple":
            tr = SimAnnealDropoutSimple(
                args.model, model, train_loader, test_loader, device,
                args.epoch, args.save_path, args.threshold
            )
            cfg = SASimpleConfig(seed=args.seed)
            trained_model, num_step = tr.train_with_sa(cfg)
            print("Number of steps:", num_step)

        elif args.mode == "gaRevisionFinal":
            ga = GaDropout(
                args.model, model, train_loader, test_loader, device,
                args.epoch, args.save_path, args.threshold, seed=args.seed
            )
            trained_model, num_step = ga.train_with_genetic()
            print("Number of steps:", num_step)

        elif args.mode == "EchoSRD_with_revision":
            # Echo-SRD (oscillating SRD) + final full-data revision
            tr = EchoSRD(
                args.model, model, train_loader, test_loader, device,
                args.epoch, args.save_path, args.threshold
            )
            cfg = EchoSRDConfig(seed=args.seed)
            # Optional CLI overrides (otherwise use EchoSRD_with_revision.py globals)
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
            print("Number of steps:", num_step)


        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    eff_epoch = int(num_step / data_size)
    print("Effective Epochs: ", eff_epoch)
    torch.save(trained_model, "trained_model.pth")
if __name__ == "__main__":
    main()
