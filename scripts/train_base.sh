#!/bin/bash

source activate aw847-torch

python -u run.py -fp rawbrain16p3_rescaleinputs --epochs 10000 --batch_size 64 --lr 1e-3 --unet_hdim 32 --hnet_hdim 64 --undersampling_rate 16p3 --losses l1 ssim --sampling uhs --no_preprocess_dataset --organ brain --no_anneal --rescale_in --hyperparameters 0
