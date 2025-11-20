#!/bin/bash

# 显卡ID
export CUDA_VISIBLE_DEVICES=0

python train_test.py \
--output_path results/baseline_market \
--train_dataset market \
--test_dataset market \
--datasets_root /home/disk5/yj/Datasets/ \
--p 16 \
--k 4 \
--total_train_epochs 40 \
--mode train