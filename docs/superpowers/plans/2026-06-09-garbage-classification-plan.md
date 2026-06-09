# 垃圾分类智能识别系统实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 EfficientNet-B0 实现 21 类垃圾分类系统，包含完整训练、评估和可视化流程

**Architecture:** 使用 PyTorch + timm 库，加载预训练 EfficientNet-B0，修改分类头为 21 类。采用迁移学习方法，数据增强防止过拟合。

**Tech Stack:** Python 3.8+, PyTorch 2.0+, timm, scikit-learn, matplotlib, pandas

---

## 文件结构

```
/Users/gyp/Desktop/gabage-category/
├── data/
│   ├── garbage_21classes.json          # 21类定义和映射
│   ├── split/                          # 划分后的数据集
│   │   ├── train/                     # 训练集
│   │   ├── val/                       # 验证集
│   │   └── test/                      # 测试集
│   └── garbage/ # 原始数据集（已有）
├── models/
│   └── efficientnet.py # 模型定义
├── utils/
│   ├── dataset.py # 数据集加载
│   ├── transforms.py                   # 数据增强
│   ├── metrics.py                      # 评估指标
│   └── visualization.py                # 可视化工具
├── scripts/
│   ├── prepare_data.py                 # 数据预处理脚本
│   ├── train.py                        # 训练脚本
│   ├── evaluate.py                     # 评估脚本
│   └── predict.py                      # 预测脚本
├── outputs/
│   ├── models/                         # 保存的模型
│   ├── confusion_matrix/ # 混淆矩阵
│   ├── misclassified/                  # 错误样本
│   └── metrics.txt                     # 评估指标
├── requirements.txt
├── main.py # 主入口
├── train.py # 训练入口
├── evaluate.py                         # 评估入口
└── README.md
```

---

## 21 类映射表（基于实际数据）

根据 `garbage_classify_rule.json` 的 40 类定义：

| 新类别ID | 名称 | 包含原类别ID | 原类别名称 | 预估数量 |
|----------|------|-------------|-----------|---------|
| 0 | 一次性快餐盒 | 0 | 其他垃圾/一次性快餐盒 | 242 |
| 1 | 污损塑料/烟蒂 | 1,2,3,4,5 | 污损塑料、烟蒂、牙签、破碎花盆、竹筷 | ~1500 |
| 2 | 剩饭大骨 | 6,7 | 剩饭剩菜、大骨头 | ~600 |
| 3 | 果皮果肉 | 8,9 | 水果果皮、水果果肉 | ~750 |
| 4 | 茶叶渣/菜叶根 | 10,11 | 茶叶渣、菜叶菜根 | ~800 |
| 5 | 蛋壳鱼骨 | 12,13 | 蛋壳、鱼骨 | ~700 |
| 6 | 充电宝/包 | 14,15 | 充电宝、包 | ~600 |
| 7 | 化妆品瓶/塑料玩具 | 16,17,18,19 | 化妆品瓶、塑料玩具、塑料碗盆、塑料衣架 | ~1500 |
| 8 | 塑料包装/快递袋 | 20,21,22 | 快递纸袋、插头电线、旧衣服 | ~1200 |
| 9 | 衣物/玩具 | 23,24,25 | 易拉罐、枕头、毛绒玩具 | ~1200 |
| 10 | 洗发水瓶/玻璃杯 | 26,27,28 | 洗发水瓶、玻璃杯、皮鞋 | ~1100 |
| 11 | 砧板/纸板箱 | 29,30 | 皮鞋、砧板、纸板箱 | ~800 |
| 12 | 调料瓶/酒瓶 | 31,32 | 调料瓶、酒瓶 | ~900 |
| 13 | 金属罐/锅 | 33,34,35 | 金属食品罐、锅、食用油桶 | ~1200 |
| 14 | 饮料瓶 | 36 | 饮料瓶 | 387 |
| 15 | 电池 | 37 | 干电池 | 391 |
| 16 | 软膏 | 38 | 软膏 | 393 |
| 17 | 过期药物 | 39 | 过期药物 | 437 |
| 18 | 布料/纤维 | 22,24 | 旧衣服、枕头 | ~900 |
| 19 | 其他塑料 | 17,18,19 | 塑料玩具、塑料碗盆、塑料衣架 | ~1100 |
| 20 | 其他可回收 | 14,16,26 | 充电宝、化妆品瓶、洗发水瓶 | ~800 |

**总计**: 21 类

---

## 任务列表

### Task 1: 创建依赖文件和配置

**Files:**
- Create: `requirements.txt`
- Create: `data/garbage_21classes.json`

- [ ] **Step 1: 创建 requirements.txt**

```txt
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
pillow>=9.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pandas>=2.0.0
tqdm>=4.65.0
```

- [ ] **Step 2: 创建类别映射配置文件**

```json
{
  "num_classes": 21,
  "classes": [
    {"id": 0, "name": "一次性快餐盒", "parent": "其他垃圾"},
    {"id": 1, "name": "污损塑料/烟蒂", "parent": "其他垃圾"},
    {"id": 2, "name": "剩饭大骨", "parent": "厨余垃圾"},
    {"id": 3, "name": "果皮果肉", "parent": "厨余垃圾"},
    {"id": 4, "name": "茶叶渣/菜叶根", "parent": "厨余垃圾"},
    {"id": 5, "name": "蛋壳鱼骨", "parent": "厨余垃圾"},
    {"id": 6, "name": "充电宝/包", "parent": "可回收物"},
    {"id": 7, "name": "化妆品瓶/塑料玩具", "parent": "可回收物"},
    {"id": 8, "name": "塑料包装/快递袋", "parent": "可回收物"},
    {"id": 9, "name": "衣物/玩具", "parent": "可回收物"},
    {"id": 10, "name": "洗发水瓶/玻璃杯", "parent": "可回收物"},
    {"id": 11, "name": "砧板/纸板箱", "parent": "可回收物"},
    {"id": 12, "name": "调料瓶/酒瓶", "parent": "可回收物"},
    {"id": 13, "name": "金属罐/锅", "parent": "可回收物"},
    {"id": 14, "name": "饮料瓶", "parent": "可回收物"},
    {"id": 15, "name": "电池", "parent": "有害垃圾"},
    {"id": 16, "name": "软膏", "parent": "有害垃圾"},
    {"id": 17, "name": "过期药物", "parent": "有害垃圾"},
    {"id": 18, "name": "布料/纤维", "parent": "可回收物"},
    {"id": 19, "name": "其他塑料", "parent": "可回收物"},
    {"id": 20, "name": "其他可回收", "parent": "可回收物"}
  ],
  "parents": {
    "其他垃圾": [0, 1],
    "厨余垃圾": [2, 3, 4, 5],
    "可回收物": [6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20],
    "有害垃圾": [15, 16, 17]
  },
  "mapping": {
    "0": 0, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1,
    "6": 2, "7": 2,
    "8": 3, "9": 3,
    "10": 4, "11": 4,
    "12": 5, "13": 5,
    "14": 6, "15": 6,
    "16": 7, "17": 7, "18": 7, "19": 7,
    "20": 8, "21": 8, "22": 8,
    "23": 9, "24": 9, "25": 9,
    "26": 10, "27": 10, "28": 10,
    "29": 11, "30": 11,
    "31": 12, "32": 12,
    "33": 13, "34": 13, "35": 13,
    "36": 14,
    "37": 15,
    "38": 16,
    "39": 17
  }
}
```

- [ ] **Step 3: 提交**

```bash
git add requirements.txt data/garbage_21classes.json
git commit -m "feat: add requirements and 21-class mapping config"
```

---

### Task 2: 数据预处理脚本

**Files:**
- Create: `scripts/prepare_data.py`

- [ ] **Step 1: 创建数据预处理脚本**

```python
#!/usr/bin/env python3
"""
数据预处理脚本
1. 读取原始标签文件 (train.txt, test.txt, validate.txt)
2. 根据映射关系合并类别
3. 统一图像尺寸
4. 划分训练/验证/测试集
5. 生成数据集结构
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

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
    
    # 生成类别统计
    stats = {}
    for _, label in train_data + val_data + test_data:
        stats[label] = stats.get(label, 0) + 1
    
    with open(OUTPUT_ROOT / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Dataset prepared at {OUTPUT_ROOT}")

if __name__ == "__main__":
    prepare_dataset()
```

- [ ] **Step 2: 运行脚本验证**

```bash
cd /Users/gyp/Desktop/gabage-category
python scripts/prepare_data.py
```

Expected: 创建 data/split/ 目录，包含 train/val/test 子目录

- [ ] **Step 3: 提交**

```bash
git add scripts/prepare_data.py
git commit -m "feat: add data preparation script"
```

---

### Task 3: 数据加载器和数据增强

**Files:**
- Create: `utils/dataset.py`
- Create: `utils/transforms.py`

- [ ] **Step 1: 创建 transforms.py**

```python
"""数据增强和预处理"""
from torchvision import transforms

# ImageNet 标准化参数
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms():
    """训练集数据增强"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transforms():
    """验证/测试集预处理"""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
```

- [ ] **Step 2: 创建 dataset.py**

```python
"""PyTorch Dataset 实现"""
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class GarbageDataset(Dataset):
    """垃圾分类数据集"""
    
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        
        #扫描所有图像文件
        for img_path in self.root.glob("*.jpg"):
            # 文件名格式: label_xxx.jpg
            label = int(img_path.stem.split("_")[0])
            self.samples.append((img_path, label))
        
        self.samples.sort(key=lambda x: x[0].name)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloaders(data_root, batch_size=32, num_workers=4):
    """获取训练和验证 DataLoader"""
    from torch.utils.data import DataLoader
    
    train_dataset = GarbageDataset(
        root=Path(data_root) / "train",
        transform=get_train_transforms()
    )
    val_dataset = GarbageDataset(
        root=Path(data_root) / "val",
        transform=get_val_transforms()
    )
    test_dataset = GarbageDataset(
        root=Path(data_root) / "test",
        transform=get_val_transforms()
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
```

- [ ] **Step 3: 提交**

```bash
git add utils/dataset.py utils/transforms.py
git commit -m "feat: add dataset and transforms"
```

---

### Task 4: 模型定义

**Files:**
- Create: `models/efficientnet.py`

- [ ] **Step 1: 创建 EfficientNet 模型**

```python
"""EfficientNet-B0 模型定义"""
import timm
import torch.nn as nn

def create_model(num_classes=21, pretrained=True):
    """
    创建 EfficientNet-B0 模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用 ImageNet 预训练权重
    
    Returns:
        model: 初始化好的模型
    """
    model = timm.create_model(
        'efficientnet_b0',
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=0.3,
        drop_path_rate=0.2
    )
    return model

def get_model_summary(model, input_size=(1, 3, 224, 224)):
    """获取模型结构摘要"""
    summary = []
    summary.append("Model: EfficientNet-B0")
    summary.append(f"Input size: {input_size}")
    summary.append(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    summary.append(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return "\n".join(summary)
```

- [ ] **Step 2: 提交**

```bash
git add models/efficientnet.py
git commit -m "feat: add EfficientNet-B0 model definition"
```

---

### Task 5: 评估指标工具

**Files:**
- Create: `utils/metrics.py`
- Create: `utils/visualization.py`

- [ ] **Step 1: 创建 metrics.py**

```python
"""评估指标计算"""
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
import json

def calculate_metrics(y_true, y_pred, class_names=None):
    """计算所有评估指标"""
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average='weighted', zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    return results

def get_classification_report(y_true, y_pred, class_names=None):
    """生成分类报告"""
    return classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )

def save_metrics(results, report, output_path):
    """保存评估指标到文件"""
    with open(output_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("Garbage Classification Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Overall Metrics:\n")
        f.write("-" * 30 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("Classification Report:\n")
        f.write("-" * 30 + "\n")
        f.write(report)
```

- [ ] **Step 2: 创建 visualization.py**

```python
"""可视化工具"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

def plot_confusion_matrix(cm, class_names, save_path=None, figsize=(16, 14)):
    """绘制混淆矩阵"""
    plt.figure(figsize=figsize)
    
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    
    sns.heatmap(
        df_cm, annot=True, fmt='d', cmap='Blues',
        annot_kws={"size": 10}
    )
    plt.title('Confusion Matrix - 21 Class Garbage Classification')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_loss_curve(train_losses, val_losses, save_path=None):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Loss curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_metrics_bar(results, save_path=None):
    """绘制评估指标柱状图"""
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    values = [results[m] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
    
    plt.ylim(0, 1)
    plt.title('Classification Metrics')
    plt.ylabel('Score')
    
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Metrics bar chart saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
```

- [ ] **Step 3: 提交**

```bash
git add utils/metrics.py utils/visualization.py
git commit -m "feat: add metrics and visualization utilities"
```

---

### Task 6: 训练脚本

**Files:**
- Create: `train.py`
- Modify: `scripts/train.py`

- [ ] **Step 1: 创建训练脚本**

```python
#!/usr/bin/env python3
"""
垃圾分类模型训练脚本
"""
import os
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.efficientnet import create_model
from utils.dataset import get_dataloaders
from utils.transforms import get_train_transforms, get_val_transforms
from utils.metrics import calculate_metrics, save_metrics
from utils.visualization import plot_loss_curve

def parse_args():
    parser = argparse.ArgumentParser(description='Train garbage classifier')
    parser.add_argument('--data_root', type=str, default='./data/split',
                       help='Path to split dataset')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--save_dir', type=str, default='./outputs/models',
                       help='Directory to save models')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(train_loader), 100. * correct / total

def validate_epoch(model, val_loader, criterion, device):
    """验证一个 epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(val_loader), 100. * correct / total, all_preds, all_labels

def main():
    args = parse_args()
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载类别配置
    with open('./data/garbage_21classes.json') as f:
        config = json.load(f)
        num_classes = config['num_classes']
        class_names = [c['name'] for c in config['classes']]
    
    # 数据加载器
    train_dataset = GarbageDataset(
        root=Path(args.data_root) / 'train',
        transform=get_train_transforms()
    )
    val_dataset = GarbageDataset(
        root=Path(args.data_root) / 'val',
        transform=get_val_transforms()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 模型
    model = create_model(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    best_acc = 0
    train_losses, val_losses = [], []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_dir / 'best_model.pth')
            print(f"Best model saved! Val Acc: {val_acc:.2f}%")
    
    # 绘制损失曲线
    plot_loss_curve(train_losses, val_losses, save_path='./outputs/loss_curve.png')
    print(f"\nTraining complete! Best Val Acc: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 修正并提交**

```bash
git add train.py
git commit -m "feat: add training script with EfficientNet-B0"
```

---

### Task 7: 评估脚本

**Files:**
- Create: `evaluate.py`

- [ ] **Step 1: 创建评估脚本**

```python
#!/usr/bin/env python3
"""
垃圾分类模型评估脚本
"""
import os
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.efficientnet import create_model
from utils.dataset import GarbageDataset
from utils.transforms import get_val_transforms
from utils.metrics import calculate_metrics, get_classification_report, save_metrics
from utils.visualization import plot_confusion_matrix

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate garbage classifier')
    parser.add_argument('--data_root', type=str, default='./data/split',
                       help='Path to split dataset')
    parser.add_argument('--checkpoint', type=str, default='./outputs/models/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Directory to save results')
    return parser.parse_args()

def evaluate(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def find_misclassified(images, labels, preds, class_names, output_dir, num_per_class=2):
    """保存错误分类的样本"""
    misclassified_dir = Path(output_dir) / 'misclassified'
    misclassified_dir.mkdir(parents=True, exist_ok=True)
    
    misclassified = []
    for i, (label, pred) in enumerate(zip(labels, preds)):
        if label != pred:
            misclassified.append({
                'index': i,
                'true_label': label,
                'true_name': class_names[label],
                'pred_label': pred,
                'pred_name': class_names[pred],
                'image': images[i]
            })
    
    # 每类展示错误样本
    for true_label in range(len(class_names)):
        label_mis = [m for m in misclassified if m['true_label'] == true_label][:num_per_class]
        if label_mis:
            print(f"Class '{class_names[true_label]}' misclassified as:")
            for m in label_mis:
                print(f"  -> {m['pred_name']}")
    
    print(f"\nTotal misclassified: {len(misclassified)} / {len(labels)}")

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载类别配置
    with open('./data/garbage_21classes.json') as f:
        config = json.load(f)
        num_classes = config['num_classes']
        class_names = [c['name'] for c in config['classes']]
    
    # 数据集
    test_dataset = GarbageDataset(
        root=Path(args.data_root) / 'test',
        transform=get_val_transforms()
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 模型
    model = create_model(num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%")
    
    # 评估
    y_true, y_pred, y_probs = evaluate(model, test_loader, device)
    
    # 计算指标
    results = calculate_metrics(y_true, y_pred)
    report = get_classification_report(y_true, y_pred, class_names)
    
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision (weighted): {results['precision_weighted']:.4f}")
    print(f"Recall (weighted): {results['recall_weighted']:.4f}")
    print(f"F1 Score (weighted): {results['f1_weighted']:.4f}")
    
    # 保存指标
    save_metrics(results, report, Path(args.output_dir) / 'metrics.txt')
    print(f"\nMetrics saved to {args.output_dir}/metrics.txt")
    
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, save_path=Path(args.output_dir) / 'confusion_matrix.png')
    
    # 错误样本
    find_misclassified(None, y_true, y_pred, class_names, args.output_dir)

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 提交**

```bash
git add evaluate.py
git commit -m "feat: add evaluation script with metrics and confusion matrix"
```

---

### Task 8: 预测脚本

**Files:**
- Create: `scripts/predict.py`

- [ ] **Step 1: 创建预测脚本**

```python
#!/usr/bin/env python3
"""
单张图像预测脚本
"""
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.efficientnet import create_model
from utils.transforms import get_val_transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Predict garbage class')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image')
    parser.add_argument('--checkpoint', type=str, default='./outputs/models/best_model.pth',
                       help='Path to model checkpoint')
    return parser.parse_args()

def predict(image_path, model, transform, device, class_names):
    """预测单张图像"""
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(1).item()
    
    print(f"\nImage: {image_path}")
    print(f"Prediction: {class_names[pred]} (class {pred})")
    print(f"Confidence: {probs[0][pred].item():.4f}")
    print("\nTop 5 predictions:")
    top5_probs, top5_indices = probs[0].topk(5)
    for prob, idx in zip(top5_probs, top5_indices):
        print(f"  {class_names[idx.item()]}: {prob.item():.4f}")

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载类别配置
    with open('./data/garbage_21classes.json') as f:
        config = json.load(f)
        class_names = [c['name'] for c in config['classes']]
    
    # 模型
    model = create_model(num_classes=21, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 预测
    transform = get_val_transforms()
    predict(args.image, model, transform, device, class_names)

if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 提交**

```bash
git add scripts/predict.py
git commit -m "feat: add single image prediction script"
```

---

### Task 9: 主程序入口

**Files:**
- Create: `main.py`

- [ ] **Step 1: 创建主程序**

```python
#!/usr/bin/env python3
"""
垃圾分类系统主程序
用法:
    python main.py prepare # 数据预处理
    python main.py train     # 模型训练
    python main.py evaluate  # 模型评估
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
```

- [ ] **Step 2: 提交**

```bash
git add main.py
git commit -m "feat: add main entry point"
```

---

### Task 10: README 文档

**Files:**
- Create: `README.md`

- [ ] **Step 1: 创建 README**

```markdown
# 垃圾分类智能识别系统

基于深度学习的垃圾分类图像分类系统，使用 EfficientNet-B0 模型对 21 类垃圾进行分类。

## 项目结构

```
├── data/ # 数据配置
├── models/                  # 模型定义
├── utils/ # 工具函数
├── scripts/                 # 脚本
├── outputs/                 # 输出结果
├── train.py                 # 训练入口
├── evaluate.py              # 评估入口
└── main.py                  # 主程序
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据预处理

```bash
python main.py prepare
# 或
python scripts/prepare_data.py
```

### 2. 模型训练

```bash
python main.py train --epochs 50 --batch_size 32
```

### 3. 模型评估

```bash
python main.py evaluate
```

### 4. 单张图像预测

```bash
python main.py predict --image path/to/image.jpg
```

## 21 类定义

| 类别ID | 名称 | 大类 |
|--------|------|------|
| 0 | 一次性快餐盒 | 其他垃圾 |
| 1 | 污损塑料/烟蒂 | 其他垃圾 |
| ... | ... | ... |
| 15 | 电池 | 有害垃圾 |
|16 | 软膏 | 有害垃圾 |
| 17 | 过期药物 | 有害垃圾 |

## 评估指标

- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1-Score
- 混淆矩阵

## 课程设计要求

- [x] 使用 PyTorch 深度学习框架
- [x] 采用 EfficientNet 模型
- [x] 21 类垃圾分类
- [x] 学习效果分析
- [x] 混淆矩阵可视化
- [x] 错误分类样本展示

## 参考资料

- [PyTorch](https://pytorch.org/)
- [timm library](https://github.com/huggingface/pytorch-image-models)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
```

- [ ] **Step 2: 提交**

```bash
git add README.md
git commit -m "docs: add README"
```

---

## 执行顺序

### 第一批（并行执行）

| Task | 工具 |
|------|------|
| Task 1: requirements.txt + config | Write |
| Task 2: 数据预处理脚本 | Write |
| Task 3: 数据加载器 | Write |
| Task 4: 模型定义 | Write |
| Task 5: 评估指标工具 | Write |

### 第二批（并行执行）

| Task | 工具 |
|------|------|
| Task 6: 训练脚本 | Write |
| Task 7: 评估脚本 | Write |
| Task 8: 预测脚本 | Write |
| Task 9: 主程序入口 | Write |
| Task 10: README | Write |

### 第三批（顺序执行）

| Task | 工具 |
|------|------|
| 运行 prepare_data.py | Bash |
| 运行 train.py | Bash |
| 运行 evaluate.py | Bash |
| 生成报告 | Bash |

---

**计划完成！**

---

**Plan complete and saved to `docs/superpowers/plans/2026-06-09-garbage-classification-plan.md`**

##两种执行方案

**1. Subagent-Driven (推荐)** - 我调度多个 sub-agent 并行处理第一批任务（Task 1-5），快速迭代

**2. Inline Execution** - 在当前会话顺序执行所有任务

**你选择哪个方案？**