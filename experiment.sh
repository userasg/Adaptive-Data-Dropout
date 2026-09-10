#!/bin/bash

#SBATCH -c8 --mem=24g
#SBATCH --gres gpu:4
#SBATCH -p cs -q csug

source /usr2/share/gpu.sbatch

python ./main.py \
  --model efficientnet_b0 \
  --mode gaRevisionFinal \
  --epoch 30 \
  --save_path cifar10_results_30epochs/efficientnet_b0 \
  --dataset cifar10 \
  --batch_size 32 \
  --start_revision 0 \
  --task classification \
  --threshold 0.3 \
  --seed 123

  #pip install timm medmnist matplotlib psutil
  #cd gdrive/MyDrive/energy_eff_training-main/training_models