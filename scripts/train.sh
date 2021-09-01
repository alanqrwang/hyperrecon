#!/bin/bash

source activate aw847-torch

python -u run.py -fp epoch1024_schedstep128 \
  --method uhs \
  --undersampling_rate 8p2 \
  --loss_list l1 ssim
