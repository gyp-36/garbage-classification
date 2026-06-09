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
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.efficientnet import create_model
from utils.dataset import GarbageDataset
from utils.transforms import get_train_transforms, get_val_transforms
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
    return parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='训练中')
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

        pbar.set_postfix({'损失': f'{loss.item():.4f}', '准确率': f'{100.*correct/total:.2f}%'})

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
        for images, labels in tqdm(val_loader, desc='验证中'):
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
    print(f"使用设备: {device}")

    # 保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载类别配置
    with open('./data/garbage_21classes.json') as f:
        config = json.load(f)
        num_classes = config['num_classes']
        class_names = [c['name'] for c in config['classes']]
        print(f"类别数: {num_classes}")

    # 数据加载器
    train_dataset = GarbageDataset(
        root=Path(args.data_root) / 'train',
        transform=get_train_transforms()
    )
    val_dataset = GarbageDataset(
        root=Path(args.data_root) / 'val',
        transform=get_val_transforms()
    )

    print(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")

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
        print(f"\n第 {epoch}/{args.epochs} 轮")
        print("-" * 40)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        print(f"学习率: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_dir / 'best_model.pth')
            print(f"最佳模型已保存! 验证准确率: {val_acc:.2f}%")

    # 绘制损失曲线
    plot_loss_curve(train_losses, val_losses, save_path='./outputs/loss_curve.png')
    print(f"\n训练完成! 最佳验证准确率: {best_acc:.2f}%")

if __name__ == '__main__':
    main()