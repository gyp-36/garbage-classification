#!/usr/bin/env python3
"""
垃圾分类模型评估脚本
"""
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import sys

sys.path.insert(0, str(Path(__file__).parent))

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

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Test samples: {len(test_dataset)}")

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
    save_metrics(results, report, output_dir / 'metrics.txt')
    print(f"\nMetrics saved to {output_dir}/metrics.txt")

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, save_path=output_dir / 'confusion_matrix.png')

    # 错误分类统计
    misclassified = y_true != y_pred
    print(f"\nMisclassified: {misclassified.sum()} / {len(y_true)} ({100*misclassified.sum()/len(y_true):.2f}%)")

    # 每类错误统计
    print("\nMisclassification by class:")
    for i, name in enumerate(class_names):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_error = (y_pred[class_mask] != i).sum()
            print(f"  {name}: {class_error}/{class_mask.sum()} errors")

if __name__ == '__main__':
    main()