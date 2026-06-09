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