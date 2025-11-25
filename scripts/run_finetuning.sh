#!/bin/bash

# 设置错误时立即退出
set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# 切换到项目根目录（scripts的父目录）
cd "$SCRIPT_DIR/.."

# 打印当前工作目录（可选，用于调试）
echo "当前工作目录: $(pwd)"

# 执行微调脚本
python src/fine_turning/fine_stru.py

# 如果上面命令失败，会打印错误信息
if [ $? -ne 0 ]; then
    echo "微调脚本执行失败！"
    exit 1
fi

echo "微调脚本执行完成。"