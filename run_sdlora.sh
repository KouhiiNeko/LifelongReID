#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# SD-LoRA 实验配置
# 1. --cnnbackbone vit_sd_lora: 使用我们刚写的类
# 2. --weight_decay: 论文没细说，但通常 PEFT 不需要太大的 decay，先用 0.0005 或 0.01 试试
# 3. --task_base_learning_rate: 论文里写 Adam lr=0.008 (Table 3 caption)，但那是 ImageNet
#    对于 ReID，我们先保持 0.00035，如果收敛慢再调大。

python train_test.py \
--output_path results/sdlora_5steps_lowKD_highDecay \
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
--mode train