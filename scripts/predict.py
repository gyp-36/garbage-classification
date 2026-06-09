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

    print(f"\n图像: {image_path}")
    print(f"预测结果: {class_names[pred]} (类别 {pred})")
    print(f"置信度: {probs[0][pred].item():.4f}")
    print("\n前5个预测:")
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