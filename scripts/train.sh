#!/bin/bash

NAME="test"
RATE="8p3"
python -u run.py \
  -fp $NAME \
  --log_interval 25 \
  --image_dims 256 256 \
  --lr 0.001 \
  --batch_size 32 \
  --num_epochs 100 \
  --arch hyperunet \
  --hnet_hdim 128 \
  --unet_hdim 32 \
  --seed 1 \
  --forward_type csmri \
  --undersampling_rate $RATE \
  --mask_type poisson \
  --loss_list l1 ssim \
  --method base_train \
  --distribution uniform  \
  --models_dir out/
  --train_path ... \
  --test_path ... 


# Baseline
# NAME="brain-rician-snr5"
# DATE="Nov_05"
# RATE="8p3"
# python -u run.py \
#   -fp $NAME \
#   --log_interval 25 \
#   --num_val_subjects 1 \
#   --lr 0.001 \
#   --batch_size 32 \
#   --num_epochs 100 \
#   --num_steps_per_epoch 256 \
#   --arch unet \
#   --seed 1 \
#   --forward_type csmri \
#   --undersampling_rate $RATE \
#   --mask_type poisson \
#   --loss_list l1 ssim \
#   --method base_train \
#   --no_unet_residual \
#   --distribution constant \
#   --hyperparameters $A