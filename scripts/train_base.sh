#!/bin/bash

source activate aw847-torch

python -u run.py -fp hypernet_knee_8p3 --log_interval 1 --lr 1e-3 --epochs 1000 --hnet_hdim 64 --losses l1 ssim --sampling uhs --range_restrict --undersampling_rate 8p3
