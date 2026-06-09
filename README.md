# 垃圾分类智能识别系统

基于深度学习的垃圾分类图像分类系统，使用 EfficientNet-B0 模型对 21 类垃圾进行分类。

## 项目结构

```
├── data/
│   ├── garbage_21classes.json    # 21类定义和映射
│   └── split/                    # 划分后的数据集
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   └── efficientnet.py           # EfficientNet-B0 模型定义
├── utils/
│   ├── dataset.py                # PyTorch Dataset
│   ├── transforms.py             # 数据增强
│   ├── metrics.py                # 评估指标
│   └── visualization.py          # 可视化工具
├── scripts/
│   ├── prepare_data.py           # 数据预处理脚本
│   └── predict.py                # 单张图像预测脚本
├── outputs/
│   ├── models/                   # 保存的模型权重
│   ├── confusion_matrix.png      # 混淆矩阵
│   ├── metrics.txt               # 评估指标
│   └── loss_curve.png            # 损失曲线
├── requirements.txt
├── main.py                       # 主程序入口
├── train.py                      # 训练入口
├── evaluate.py                   # 评估入口
└── README.md
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
| 2 | 剩饭大骨 | 厨余垃圾 |
| 3 | 果皮果肉 | 厨余垃圾 |
| 4 | 茶叶渣/菜叶根 | 厨余垃圾 |
| 5 | 蛋壳鱼骨 | 厨余垃圾 |
| 6 | 充电宝/包 | 可回收物 |
| 7 | 化妆品瓶/塑料玩具 | 可回收物 |
| 8 | 塑料包装/快递袋 | 可回收物 |
| 9 | 衣物/玩具 | 可回收物 |
| 10 | 洗发水瓶/玻璃杯 | 可回收物 |
| 11 | 砧板/纸板箱 | 可回收物 |
| 12 | 调料瓶/酒瓶 | 可回收物 |
| 13 | 金属罐/锅 | 可回收物 |
| 14 | 饮料瓶 | 可回收物 |
| 15 | 电池 | 有害垃圾 |
| 16 | 软膏 | 有害垃圾 |
| 17 | 过期药物 | 有害垃圾 |
| 18 | 布料/纤维 | 可回收物 |
| 19 | 其他塑料 | 可回收物 |
| 20 | 其他可回收 | 可回收物 |

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

## 技术栈

- Python 3.8+
- PyTorch 2.0+
- timm (PyTorch Image Models)
- scikit-learn
- matplotlib
- pandas

## 参考资料

- [PyTorch](https://pytorch.org/)
- [timm library](https://github.com/huggingface/pytorch-image-models)
- [EfficientNet](https://arxiv.org/abs/1905.11946)