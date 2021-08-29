#!/bin/bash

mkdir -p job_err
mkdir -p job_out

for i in 0.0 0.25 0.5 0.75 1.0
do
  sbatch --export=ALL,A=$i --requeue -p sablab -t 48:00:00 --mem=8G --gres=gpu:1 --job-name=base$i -e ./job_err/%j-base$i.err -o ./job_out/%j-base$i.out $1
done

