#!/bin/bash
# ZeRO 1/2/3 显存对比测试
#
# 使用方法:
#   1. 打开另一个终端运行: watch -n 1 nvidia-smi
#   2. 运行本脚本，记录每个配置的显存占用

cd "$(dirname "$0")"

echo "============================================"
echo "ZeRO 显存对比测试"
echo "============================================"
echo ""
echo "请在另一个终端运行: watch -n 1 nvidia-smi"
echo "观察并记录每个阶段的显存占用"
echo ""

# ZeRO-1
echo ">>> 测试 ZeRO-1 <<<"
ACCELERATE_CONFIG_FILE=accelerate_zero1.yaml \
accelerate launch --config_file accelerate_zero1.yaml train.py
echo ""
echo "ZeRO-1 完成，请记录显存占用"
echo "按回车继续..."
read

# ZeRO-2
echo ">>> 测试 ZeRO-2 <<<"
ACCELERATE_CONFIG_FILE=accelerate_zero2.yaml \
accelerate launch --config_file accelerate_zero2.yaml train.py
echo ""
echo "ZeRO-2 完成，请记录显存占用"
echo "按回车继续..."
read

# ZeRO-3
echo ">>> 测试 ZeRO-3 <<<"
ACCELERATE_CONFIG_FILE=accelerate_zero3.yaml \
accelerate launch --config_file accelerate_zero3.yaml train.py
echo ""
echo "ZeRO-3 完成，请记录显存占用"

echo ""
echo "============================================"
echo "测试完成！请对比三个阶段的显存占用"
echo "============================================"
