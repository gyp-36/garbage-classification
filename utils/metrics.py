"""评估指标计算"""
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)

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
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write("垃圾分类评估结果\n")
        f.write("=" * 50 + "\n\n")

        f.write("总体指标:\n")
        f.write("-" * 30 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("分类详细报告:\n")
        f.write("-" * 30 + "\n")
        f.write(report)