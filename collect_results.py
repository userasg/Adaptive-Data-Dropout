"""
collect_results.py - assemble an ADD Table 1 style grid from completed runs.

Accuracy comes from the "<save_path>_test" JSON written by
utils.plot_accuracy_time_multi_test; Effective Epochs is parsed out of the run
log, because EE is printed but never stored in the JSON.

Usage:
    python collect_results.py                      # defaults to cifar10_repro
    python collect_results.py cifar10_results_30epochs
"""
import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "cifar10_repro"

MODELS = ["efficientnet_b0", "mobilenet_v2", "resnet50", "efficientformer"]
METHODS = [("baseline", "Baseline"),
           ("dbpd", "DBPD"),
           ("adaptive_alpha", "Adaptive-a"),
           ("adaptive_t", "Adaptive-T")]

# ADD Table 1, CIFAR-10 block: accuracy / effective epochs
PAPER = {
    "efficientnet_b0":  {"baseline": (88.21, 30), "dbpd": (87.82, 5.86),
                         "adaptive_alpha": (85.69, 7.45), "adaptive_t": (85.28, 6.36)},
    "mobilenet_v2":     {"baseline": (85.49, 30), "dbpd": (85.39, 6.42),
                         "adaptive_alpha": (83.08, 8.76), "adaptive_t": (80.79, 6.14)},
    "resnet50":         {"baseline": (84.85, 30), "dbpd": (85.61, 7.08),
                         "adaptive_alpha": (83.82, 8.48), "adaptive_t": (82.64, 6.24)},
    "efficientformer":  {"baseline": (80.32, 30), "dbpd": (81.17, 5.82),
                         "adaptive_alpha": (80.75, 7.21), "adaptive_t": (80.67, 6.90)},
}

EE_RE = re.compile(r"Effective Epochs:\s*([0-9.]+)")


def best_accuracy(test_json_path):
    """Highest test accuracy across all runs recorded in this JSON file."""
    if not os.path.isfile(test_json_path):
        return None
    try:
        data = json.load(open(test_json_path))
    except (json.JSONDecodeError, OSError):
        return None
    best = None
    for run in data.values():
        acc = run.get("accuracy") or []
        if acc:
            m = max(acc) * 100.0
            best = m if best is None else max(best, m)
    return best


def effective_epochs(log_path):
    if not os.path.isfile(log_path):
        return None
    hits = EE_RE.findall(open(log_path, errors="ignore").read())
    return float(hits[-1]) if hits else None


def main():
    print(f"\nResults under: {ROOT}/")
    print("cell = accuracy% / effective-epochs      (paper value in parentheses)\n")

    header = f"{'Model':20s}" + "".join(f"{label:>26s}" for _, label in METHODS)
    print(header)
    print("-" * len(header))

    for model in MODELS:
        row = f"{model:20s}"
        for key, _ in METHODS:
            acc = best_accuracy(os.path.join(ROOT, key, f"{model}_test"))
            ee = effective_epochs(os.path.join(ROOT, "logs", f"{model}_{key}.log"))
            ref = PAPER.get(model, {}).get(key)

            if acc is None and ee is None:
                cell = "-"
            else:
                a = f"{acc:.2f}" if acc is not None else "?"
                e = f"{ee:.2f}" if ee is not None else "?"
                cell = f"{a}/{e}"
            if ref:
                cell += f" ({ref[0]}/{ref[1]})"
            row += f"{cell:>26s}"
        print(row)

    print("\nMissing cells show '-'. Re-run with ./repro_add_table1_cifar10.sh")


if __name__ == "__main__":
    main()
