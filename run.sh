#!/usr/bin/env bash

gpu=0;  # GPU ID.
dataset=mnistzc;  # datset: sin, step, abs, linear, poly2d, poly3d, mnistz, mnistc, mnistzc
rep_dim=16;  # representation dimension of IV and other variables
seed=0;  # random seed

python main.py --gpu ${gpu} --dataset ${dataset} --rep_dim ${rep_dim} --seed ${seed};
