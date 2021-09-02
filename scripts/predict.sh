#!/bin/bash

source activate aw847-torch
NAME="epoch1024_schedstep128_epi"

python -u run.py -fp $NAME \
  --method uhs \
  --undersampling_rate 8p2 \
  --loss_list l1 ssim \
  --num_epochs 0 \
  --mask_type epi \
  --date Sep_01; 


for i in 0.0 0.25 0.5 0.75 1.0;
do
python -u run.py -fp $NAME \
  --method baseline \
  --undersampling_rate 8p2 \
  --loss_list l1 ssim \
  --hyperparameters $i \
  --seed 1 \
  --num_epochs 0 \
  --mask_type epi \
  --date Sep_01; 
done
