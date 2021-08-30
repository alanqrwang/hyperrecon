#!/bin/bash

source activate aw847-torch

python -u run.py -fp no_scheduler \
  --method baseline \
  --batch_size 32 \
  --lr 1e-3 \
  --unet_hdim 32 \
  --hnet_hdim 64 \
  --undersampling_rate 16p3 \
  --loss_list l1 ssim \
  --hyperparameters $A \
  --scheduler_gamma 1

