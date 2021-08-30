#!/bin/bash

source activate aw847-torch

python -u run.py -fp separate_by_subjects_test \
  --method uhs \
  --batch_size 32 \
  --lr 1e-3 \
  --unet_hdim 32 \
  --hnet_hdim 64 \
  --undersampling_rate 16p3 \
  --loss_list l1 ssim 
