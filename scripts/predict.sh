#!/bin/bash

source activate aw847-torch
NAME="epoch1024_schedstep128"

python -u run.py -fp $NAME \
  --method uhs \
  --undersampling_rate 8p2 \
  --loss_list l1 ssim \
  --num_epochs 0 \
  --date Aug_31; 


for i in 0.0 0.25 0.5 0.75 1.0;
do
python -u run.py -fp $NAME \
  --method baseline \
  --undersampling_rate 8p2 \
  --loss_list l1 ssim \
  --hyperparameters $i \
  --seed 1 \
  --num_epochs 0 \
  --date Aug_31; 
done
