#!/bin/bash

source activate aw847-torch14

python -u run.py -fp l1-fix-slice-nobn --log_interval 25 --lr 1e-05 --epochs 5000 --hnet_hdim 64 --reg_types cap tv --sampling uhs --range_restrict --no_use_bn
