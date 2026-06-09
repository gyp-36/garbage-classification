#!/usr/bin/env python3
"""
垃圾分类系统主程序
用法:
    python main.py prepare # 数据预处理
    python main.py train   # 模型训练
    python main.py evaluate # 模型评估
    python main.py predict --image path/to/image.jpg  # 预测
"""

import sys
import subprocess

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == 'prepare':
        print("Running data preparation...")
        subprocess.run([sys.executable, 'scripts/prepare_data.py'])

    elif command == 'train':
        print("Starting training...")
        subprocess.run([sys.executable, 'train.py', *sys.argv[2:]])

    elif command == 'evaluate':
        print("Running evaluation...")
        subprocess.run([sys.executable, 'evaluate.py', *sys.argv[2:]])

    elif command == 'predict':
        print("Running prediction...")
        subprocess.run([sys.executable, 'scripts/predict.py', *sys.argv[2:]])

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

if __name__ == '__main__':
    main()