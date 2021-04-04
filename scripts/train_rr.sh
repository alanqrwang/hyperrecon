#!/bin/bash

source activate aw847-torch14

python -u run.py -fp l1-fix --log_interval 25 --lr 1e-03 --epochs 5000 --hnet_hdim 64 --reg_types cap tv --sampling uhs --range_restrict --cont 3000 --date Mar_30 --force_lr 1e-4
