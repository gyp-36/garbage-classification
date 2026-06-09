"""PyTorch Dataset 实现"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class GarbageDataset(Dataset):
    """垃圾分类数据集"""

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        # 扫描所有图像文件
        for img_path in sorted(self.root.glob("*.jpg")):
            # 文件名格式: label_xxx.jpg
            try:
                label = int(img_path.stem.split("_")[0])
                self.samples.append((img_path, label))
            except ValueError:
                continue

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
    from utils.transforms import get_train_transforms, get_val_transforms

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