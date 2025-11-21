#!/bin/bash

# 显卡ID
export CUDA_VISIBLE_DEVICES=0

# 核心修改：删掉了 --task_base_learning_rate 0.005
# 让它使用默认的 0.00035

python train_test.py \
--output_path results/vit_market_fix \
--cnnbackbone vit_base \
--train_dataset market \
--test_dataset market \
--datasets_root /home/disk5/yj/Datasets \
--p 16 \
--k 4 \
--total_train_epochs 40 \
--mode train