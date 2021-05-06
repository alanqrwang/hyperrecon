#!/bin/bash

source activate aw847-torch14

python -u run.py -fp testing_losses --log_interval 25 --lr 1e-03 --epochs 5000 --hnet_hdim 64 --losses dc tv --sampling uhs --range_restrict --n_ch_out 1 --cont 1000 --force_lr 1e-5 --date Apr_14
