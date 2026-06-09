"""EfficientNet-B0 模型定义"""
import timm

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