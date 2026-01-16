#!/bin/bash
# DeepSpeed 原生启动方式
#
# 参数说明:
#   --include localhost:6,7  指定使用 GPU 6 和 7
#   --master_port 29500      分布式通信端口（多任务时需要改）

deepspeed \
    --include localhost:6,7 \
    --master_port 29500 \
    train.py

# 单卡运行:
# deepspeed --include localhost:6 train.py

# 查看显存:
# watch -n 1 nvidia-smi
