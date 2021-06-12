#!/bin/bash

source activate aw847-torch

python -u run.py -fp 50subs --epochs 10000 --batch_size 64 --lr 1e-3 --unet_hdim 32 --hnet_hdim 64 --undersampling_rate 16p3 --losses l1 ssim --sampling uhs --no_rescale_in  --no_legacy_dataset --cont 8125 --date May_26 --hyperparameters 0.75

