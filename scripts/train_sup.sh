#!/bin/bash

source activate aw847-torch

python run.py -fp stringify_loss_prefix \
  --num_epochs 10000 \
  --batch_size 32 \
  --lr 1e-3 \
  --unet_hdim 32 \
  --hnet_hdim 64 \
  --undersampling_rate 16p3 \
  --loss_list l1 ssim \
  --method uhs
