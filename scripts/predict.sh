#!/bin/bash

source activate aw847-torch
NAME="example"

python -u run.py -fp $NAME \
  --method uhs \
  --undersampling_rate 4 \
  --loss_list l1 ssim \
  --num_epochs 0 \
  --mask_type epi_vertical \
  --seed 1; 


for i in 0.0 0.25 0.5 0.75 1.0;
do
python -u run.py -fp $NAME \
  --method baseline \
  --undersampling_rate 4 \
  --loss_list l1 ssim \
  --hyperparameters $i \
  --seed 1 \
  --num_epochs 0 \
  --mask_type epi_vertical; 
done
