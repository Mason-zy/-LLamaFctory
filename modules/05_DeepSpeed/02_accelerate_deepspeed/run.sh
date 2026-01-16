#!/bin/bash
# Accelerate + DeepSpeed 启动方式
#
# 配置文件关系:
#   accelerate_config.yaml
#       └── deepspeed_config_file: ds_config.json

accelerate launch \
    --config_file accelerate_config.yaml \
    train.py

# 或者用 deepspeed 直接启动（效果一样）:
# deepspeed --include localhost:6,7 train.py --deepspeed_config ds_config.json
