"""
PyTorch U-Net V3 训练脚本
基于Keras架构转换的PyTorch实现
使用CelebAMask-HQ数据集进行训练
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import skimage.io as io
import skimage.transform as trans
import glob
from torchvision import transforms

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_pytorch_v3 import UNetV3

print("=" * 70)
print("     PyTorch U-Net V3 Training")
print("=" * 70)
print()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据集类
class CelebAMaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_images=100, target_size=(256, 256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.img_indices = list(range(num_images))
    
    def __len__(self):
        return len(self.img_indices)
    
    def __getitem__(self, idx):
        img_idx = self.img_indices[idx]
        
        # 加载图片
        img_path = os.path.join(self.img_dir, f"{img_idx}.jpg")
        img = io.imread(img_path)
        
        # 加载所有部位的掩码并合并
        mask_parts = []
        subfolder = str(img_idx // 2000)  # 每2000张图片一个文件夹
        mask_subdir = os.path.join(self.mask_dir, subfolder)
        
        # 查找所有掩码文件
        mask_files = glob.glob(os.path.join(mask_subdir, f"{img_idx:05d}_*.png"))
        
        if mask_files:
            mask_parts = []
            for mask_file in mask_files:
                # 读取为灰度图
                mask = io.imread(mask_file, as_gray=True)
                # 确保掩码是0-1范围
                if mask.max() > 1.0:
                    mask = mask / 255.0
                mask_parts.append(mask)
            
            # 合并所有部位为一个掩码（参考Keras版本）
            mask = np.max(np.stack(mask_parts), axis=0)
        else:
            # 如果没有掩码，创建全黑掩码
            mask = np.zeros_like(img[:, :, 0])
        
        # 预处理
        # 归一化
        img = img / 255.0
        
        # 调整大小（图片是1024x1024，掩码是512x512）
        img = trans.resize(img, self.target_size)
        mask = trans.resize(mask, self.target_size)
        
        # 转换为PyTorch张量
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # (H, W, C) -> (C, H, W)
        mask = torch.from_numpy(mask).float().unsqueeze(0)      # 添加通道维度
        
        return img, mask

# 设置数据集路径
celeb_root = '../CelebAMask-HQ'
img_dir = os.path.join(celeb_root, 'CelebA-HQ-img')
mask_dir = os.path.join(celeb_root, 'CelebAMask-HQ-mask-anno')

print("[1/4] Setting up dataset...")

# 检查数据是否存在
if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
    print(f"✗ Dataset not found!")
    print(f"  Image dir: {img_dir}")
    print(f"  Mask dir: {mask_dir}")
    sys.exit(1)

# 计算训练图片数量
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
num_images = len(img_files)

print(f"  Found {num_images} training images")

if num_images == 0:
    print("✗ No training data found!")
    sys.exit(1)

# 使用500张图片训练
max_train_images = 500
if num_images > max_train_images:
    print(f"  Using first {max_train_images} images")
    num_images = max_train_images

# 创建数据集和数据加载器
dataset = CelebAMaskDataset(
    img_dir=img_dir,
    mask_dir=mask_dir,
    num_images=num_images,
    target_size=(256, 256)
)

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0  # Windows环境下设置为0
)

print("✓ Dataset ready")
print()

print("[2/4] Creating model...")

# 创建模型
model = UNetV3(in_channels=3, out_channels=1).to(device)

# 定义优化器和损失函数（参考Keras版本使用1e-3学习率）
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
print()

# 创建保存目录
os.makedirs('saved_models', exist_ok=True)

print("[3/4] Starting training...")

# 训练参数（20轮）
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 40)
    
    for batch_idx, (imgs, masks) in enumerate(dataloader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # 前向传播
        outputs = model(imgs)
        
        # 计算损失
        loss = criterion(outputs, masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        preds = (outputs > 0.5).float()
        acc = (preds == masks).float().mean()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
            
            # 调试信息：检查掩码统计
            if batch_idx == 0:
                mask_sum = masks.sum().item()
                mask_mean = masks.mean().item()
                mask_min = masks.min().item()
                mask_max = masks.max().item()
                output_mean = outputs.mean().item()
                print(f"  Mask stats: sum={mask_sum:.2f}, mean={mask_mean:.4f}, min={mask_min:.4f}, max={mask_max:.4f}")
                print(f"  Output stats: mean={output_mean:.4f}")
    
    # 计算平均损失和准确率
    epoch_loss /= len(dataloader)
    epoch_acc /= len(dataloader)
    
    print(f"\nEpoch {epoch + 1} - Average Loss: {epoch_loss:.4f}, Average Acc: {epoch_acc:.4f}")
    
    # 记录训练进度
    print(f"Epoch {epoch + 1} completed")
    
    # 保存模型
    if (epoch + 1) % 1 == 0:
        model_path = os.path.join('saved_models', f'pytorch_unet_v3_epoch{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, model_path)
        print(f"✓ Model saved to: {model_path}")

# 保存最终模型
final_model_path = os.path.join('saved_models', 'pytorch_unet_v3_final.pth')
torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, final_model_path)

print()
print("=" * 70)
print("✓ Training complete!")
print("=" * 70)
print()
print(f"Final model saved as: {final_model_path}")
print()
print("Next step: Run predict_pytorch_v3.py to test the model")
print()
