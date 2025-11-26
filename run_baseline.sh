#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# ResNet50 是代码默认的 backbone，不需要指定 --cnnbackbone
# 使用默认参数 (lr=0.00035, weight_decay=0.0005)

python train_test.py \
--output_path results/lreid_resnet_3steps \
--datasets_root /home/disk5/yj/Datasets \
--train_dataset market subcuhksysu duke msmt17 cuhk03 \
--test_dataset market subcuhksysu duke msmt17 cuhk03 \
--continual_step task \
--p 16 \
--k 4 \
--total_train_epochs 40 \
--mode train