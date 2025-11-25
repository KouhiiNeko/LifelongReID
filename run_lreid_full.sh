#!/bin/bash

# 显卡ID
export CUDA_VISIBLE_DEVICES=0

# ====================================================
# LReID 5-Step Order
# 1. Market-1501
# 2. CUHK-SYSU (代码中叫 subcuhksysu)
# 3. DukeMTMC-reID
# 4. MSMT17-V2
# 5. CUHK03
# ====================================================

python train_test.py \
--output_path results/lreid_5steps_vit_fix \
--cnnbackbone vit_base \
--datasets_root /home/disk5/yj/Datasets \
--train_dataset market subcuhksysu duke msmt17 cuhk03 \
--test_dataset market subcuhksysu duke msmt17 cuhk03 \
--continual_step task \
--p 16 \
--k 4 \
--total_train_epochs 40 \
--task_base_learning_rate 0.00035 \
--weight_decay 0.05 \
--mode train