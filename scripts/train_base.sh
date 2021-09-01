#!/bin/bash

source activate aw847-torch

python -u run.py -fp epoch1024_schedstep128 \
  --method baseline \
  --undersampling_rate 16p3 \
  --loss_list l1 ssim \
  --hyperparameters $A \
  --seed 1

