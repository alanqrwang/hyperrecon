#!/bin/bash

source activate aw847-torch14

python run.py -fp 1chan --log_interval 25 --lr 1e-3 --epochs 5000 --reg_types cap tv --sampling uhs --range_restrict --hnet_hdim 64 --no_use_tanh --lr 1e-4 --cont 1850 --date Apr_05

