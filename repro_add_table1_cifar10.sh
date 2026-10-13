#!/bin/bash
# =============================================================================
# Reproduce the CIFAR-10 block of ADD Table 1 (paper/ADD.pdf)
#
#                      Baseline      DBPD        Adaptive-a    Adaptive-T
#   EfficientNet-B0    88.21 / 30    87.82/5.86  85.69/7.45    85.28/6.36
#   MobileNet-V2       85.49 / 30    85.39/6.42  83.08/8.76    80.79/6.14
#   ResNet-50          84.85 / 30    85.61/7.08  83.82/8.48    82.64/6.24
#   EfficientFormer-L1 80.32 / 30    81.17/5.82  80.75/7.21    80.67/6.90
#                                    (accuracy % / effective epochs)
#
# Settings follow PDD S5.2: CIFAR-10 = 30 epochs, AdamW, StepLR, batch size 32,
# lr 3e-4, tau = 0.3.
#
# REVISION: --final_revision y (the default) makes the LAST epoch train on the
# full dataset for every variant - the paper'"'"'s revision pass, D_T = D_0.
# Use --final_revision n to switch the revision phase off entirely.
#
# Each method writes to its own --save_path because DBPD and SMRD share a JSON
# key (no variant suffix) and would otherwise overwrite one another.
#
# Usage:   ./repro_add_table1_cifar10.sh            # all 16 runs
#          ./repro_add_table1_cifar10.sh resnet50   # one model only
# =============================================================================

set -u
cd "$(dirname "$0")"

EPOCHS=30
BATCH=32
THRESH=0.3
SEED=123
DATASET=cifar10
ROOT=cifar10_repro
LOGDIR="$ROOT/logs"

MODELS=("efficientnet_b0" "mobilenet_v2" "resnet50" "efficientformer")
if [ "$#" -gt 0 ]; then MODELS=("$@"); fi

mkdir -p "$LOGDIR"

run () {            # run <tag> <mode> <extra flags...>
  local tag="$1"; local mode="$2"; shift 2
  local log="$LOGDIR/${MODEL}_${tag}.log"
  echo ""
  echo "=== $MODEL / $tag  (--mode $mode) ==="
  echo "    log: $log"
  python ./main.py \
    --model "$MODEL" \
    --mode "$mode" \
    --epoch "$EPOCHS" \
    --dataset "$DATASET" \
    --batch_size "$BATCH" \
    --threshold "$THRESH" \
    --task classification \
    --final_revision y \
    --seed "$SEED" \
    --save_path "$ROOT/$tag/$MODEL" \
    "$@" > "$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "    FAILED (exit $rc) - last lines:"
    tail -n 15 "$log" | sed 's/^/      /'
  else
    grep -E "Effective Epochs|Test Accuracy" "$log" | tail -n 2 | sed 's/^/    /'
  fi
}

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "#############################################################"
  echo "# $MODEL"
  echo "#############################################################"

  # Baseline - EE = 30 by definition
  run baseline        baseline

  # PDD Variant 1: DBPD, tau = 0.3, final epoch is the revision pass
  run dbpd            pdd_dbpd

  # ADD Variant 1: Adaptive-alpha   (add/adaptive_alpha.py)
  run adaptive_alpha  add_adaptive_alpha

  # ADD Variant 2: Adaptive-T       (add/adaptive_keep_fraction.py)
  run adaptive_t      add_adaptive_t
done

echo ""
echo "============================================================="
echo "Done. Collect the table with:"
echo "  python collect_results.py"
echo "============================================================="
