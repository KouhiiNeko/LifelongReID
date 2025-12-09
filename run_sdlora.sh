#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 核心修改：
# 1. --weight_r 0: 彻底关闭 AKA 的图关联损失 (Relation Loss)
# 2. --weight_kd 0.1: 保留基础蒸馏 (LwF)，配合 SD-LoRA 使用
# 3. --cnnbackbone vit_sd_lora: 使用你的 SD-LoRA

python train_test.py \
--output_path results/sdlora_pure_vit \
--cnnbackbone vit_sd_lora \
--datasets_root /home/disk5/yj/Datasets \
--train_dataset market subcuhksysu duke msmt17 cuhk03 \
--test_dataset market subcuhksysu duke msmt17 cuhk03 \
--continual_step task \
--p 16 \
--k 4 \
--total_train_epochs 40 \
--task_base_learning_rate 0.00035 \
--weight_decay 0.05 \
--weight_kd 0.1 \
--weight_r 0 \
--mode train