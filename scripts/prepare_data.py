#!/usr/bin/env python3
"""
数据预处理脚本
1. 读取原始标签文件 (train.txt, test.txt, validate.txt)
2. 根据映射关系合并类别
3. 统一图像尺寸
4. 生成数据集结构
"""

import json
from pathlib import Path
from PIL import Image

# 配置
DATA_ROOT = Path("/Users/gyp/Desktop/gabage-category/garbage")
OUTPUT_ROOT = Path("/Users/gyp/Desktop/gabage-category/data/split")
IMAGE_SIZE = (224, 224)

# 类别映射文件
with open("/Users/gyp/Desktop/gabage-category/data/garbage_21classes.json") as f:
    config = json.load(f)
    CLASS_MAPPING = {int(k): v for k, v in config["mapping"].items()}

def load_label_file(filepath):
    """加载标签文件"""
    data = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_path = parts[0]
                orig_label = int(parts[1])
                new_label = CLASS_MAPPING[orig_label]
                data.append((img_path, new_label))
    return data

def process_image(src_path, dst_path):
    """处理图像：resize到统一尺寸"""
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.BILINEAR)
        img.save(dst_path)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False

def prepare_dataset():
    """准备数据集"""
    # 创建输出目录
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / split).mkdir(parents=True, exist_ok=True)

    # 加载标签
    train_data = load_label_file(DATA_ROOT / "train.txt")
    test_data = load_label_file(DATA_ROOT / "test.txt")
    val_data = load_label_file(DATA_ROOT / "validate.txt")

    print(f"Loaded: train={len(train_data)}, test={len(test_data)}, val={len(val_data)}")

    # 处理每个数据集
    for split_name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        for i, (img_path, label) in enumerate(data):
            src = DATA_ROOT / img_path
            dst = OUTPUT_ROOT / split_name / f"{label}_{i}.jpg"
            process_image(src, dst)
        print(f"Processed {split_name}: {len(data)} images")

    # 生成类别统计
    stats = {}
    for _, label in train_data + val_data + test_data:
        stats[label] = stats.get(label, 0) + 1

    with open(OUTPUT_ROOT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Dataset prepared at {OUTPUT_ROOT}")

if __name__ == "__main__":
    prepare_dataset()