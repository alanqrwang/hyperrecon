#!/bin/bash

source activate aw847-torch

python -u run.py -fp epoch1024_schedstep128_vert \
  --method uhs \
  --undersampling_rate 4 \
  --mask_type epi_vertical \
  --loss_list l1 ssim \
  --seed 1
