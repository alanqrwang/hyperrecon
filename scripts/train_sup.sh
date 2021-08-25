#!/bin/bash

source activate aw847-torch

python -u run.py -fp hypernet_base_8 --log_interval 25 --lr 1e-3 --epochs 10000 --hnet_hdim 64 --losses l1 l1 --sampling uhs --range_restrict --no_use_tanh --unet_hdim 32 --cont 3225 --date Apr_26 --force_lr 1e-5
