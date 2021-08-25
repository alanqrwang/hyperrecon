#!/bin/bash

source activate aw847-torch14

python -u run.py -fp 1chan_dhs --log_interval 25 --lr 1e-05 --epochs 5000 --hnet_hdim 64 --reg_types cap tv --sampling dhs --no_range_restrict --n_ch_out 1 --topK 8 --load="/share/sablab/nfs02/users/aw847/models/HyperRecon/1chan_v2/Apr_09/0.001_32_['cap', 'tv']_64_64_None_False_None/checkpoints/model.5000.h5"
