"""
PyTorch U-Net V3 训练脚本（优化版）
基于Keras架构转换的PyTorch实现
使用CelebAMask-HQ数据集进行训练
优化内容：数据增强、学习率调度、验证集、早停机制、最佳模型保存
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import skimage.io as io
import skimage.transform as trans
import glob
from torchvision import transforms
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_pytorch_v3 import UNetV3

print("=" * 70)
print("     PyTorch U-Net V3 Training (Optimized)")
print("=" * 70)
print()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据增强类
class DataAugmentation:
    def __init__(self):
        self.transforms = [
            self.random_flip,
            self.random_rotate,
            self.random_brightness_contrast,
            self.random_gaussian_noise
        ]
    
    def random_flip(self, img, mask):
        """随机水平翻转"""
        if random.random() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        return img, mask
    
    def random_rotate(self, img, mask):
        """随机旋转（90度倍数）"""
        if random.random() > 0.5:
            k = random.choice([1, 3])  # 90度或270度
            img = np.rot90(img, k)
            mask = np.rot90(mask, k)
        return img, mask
    
    def random_brightness_contrast(self, img, mask):
        """随机调整亮度和对比度"""
        if random.random() > 0.5:
            # 亮度调整
            brightness_factor = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness_factor, 0, 1)
            
            # 对比度调整
            contrast_factor = random.uniform(0.8, 1.2)
            img = np.clip((img - 0.5) * contrast_factor + 0.5, 0, 1)
        return img, mask
    
    def random_gaussian_noise(self, img, mask):
        """添加随机高斯噪声"""
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)
        return img, mask
    
    def apply(self, img, mask):
        """应用所有数据增强"""
        for transform in self.transforms:
            img, mask = transform(img, mask)
        return img, mask

# 数据集类（优化版）
class CelebAMaskDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_images=100, target_size=(256, 256), augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.augment = augment
        self.data_aug = DataAugmentation()
        
        # 获取所有图片文件
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        if num_images > 0:
            self.img_files = self.img_files[:num_images]
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 获取图片索引
        img_filename = self.img_files[idx]
        img_idx = int(os.path.splitext(img_filename)[0])
        
        # 加载图片
        img_path = os.path.join(self.img_dir, img_filename)
        img = io.imread(img_path)
        
        # 加载所有部位的掩码并合并
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
        
        # 应用数据增强（仅训练集）
        if self.augment:
            img, mask = self.data_aug.apply(img, mask)
        
        # 转换为PyTorch张量（先复制数组以避免负stride问题）
        img = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()  # (H, W, C) -> (C, H, W)
        mask = torch.from_numpy(mask.copy()).float().unsqueeze(0)      # 添加通道维度
        
        return img, mask

# 设置数据集路径
celeb_root = '../CelebAMask-HQ'
img_dir = os.path.join(celeb_root, 'CelebA-HQ-img')
mask_dir = os.path.join(celeb_root, 'CelebAMask-HQ-mask-anno')

print("[1/5] Setting up dataset...")

# 检查数据是否存在
if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
    print(f"✗ Dataset not found!")
    print(f"  Image dir: {img_dir}")
    print(f"  Mask dir: {mask_dir}")
    sys.exit(1)

# 计算训练图片数量
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
total_images = len(img_files)

print(f"  Found {total_images} training images")

if total_images == 0:
    print("✗ No training data found!")
    sys.exit(1)

# 使用更多图片进行训练（增加数据量）
max_train_images = 2000  # 从500增加到2000张
if total_images > max_train_images:
    print(f"  Using first {max_train_images} images")
    total_images = max_train_images

# 划分训练集和验证集（8:2比例）
train_size = int(0.8 * total_images)
val_size = total_images - train_size

# 创建数据集
train_dataset = CelebAMaskDataset(
    img_dir=img_dir,
    mask_dir=mask_dir,
    num_images=total_images,
    target_size=(256, 256),
    augment=True  # 训练集使用数据增强
)

# 分割数据集
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 4  # 增加批次大小（从2增加到4）
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0  # Windows环境下设置为0
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print(f"Dataset ready")
print(f"  Training set: {len(train_dataset)} images")
print(f"  Validation set: {len(val_dataset)} images")
print()

print("[2/5] Creating model...")

# 创建模型
model = UNetV3(in_channels=3, out_channels=1).to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 降低学习率（从1e-3到1e-4）
criterion = nn.BCELoss()

# 添加学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
print()

# 创建保存目录
os.makedirs('saved_models', exist_ok=True)

print("[3/5] Starting training...")

# 训练参数
num_epochs = 50  # 增加训练轮数（从20增加到50）
best_val_loss = float('inf')
patience = 10  # 早停耐心值
early_stop_counter = 0

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 50)
    
    # 训练循环
    for batch_idx, (imgs, masks) in enumerate(train_dataloader):
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
        
        train_loss += loss.item()
        train_acc += acc.item()
        
        # 打印进度
        if (batch_idx + 1) % 20 == 0:  # 减少打印频率
            print(f"Train Batch {batch_idx + 1}/{len(train_dataloader)} - Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")
    
    # 计算平均训练损失和准确率
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(val_dataloader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            
            preds = (outputs > 0.5).float()
            acc = (preds == masks).float().mean()
            
            val_loss += loss.item()
            val_acc += acc.item()
    
    # 计算平均验证损失和准确率
    val_loss /= len(val_dataloader)
    val_acc /= len(val_dataloader)
    
    # 更新学习率
    scheduler.step(val_loss)
    
    # 打印epoch结果
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_path = os.path.join('saved_models', 'pytorch_unet_v3_best.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'accuracy': val_acc,
        }, best_model_path)
        print(f"Best model saved to: {best_model_path}")
        early_stop_counter = 0  # 重置早停计数器
    else:
        early_stop_counter += 1
        print(f"Early stop counter: {early_stop_counter}/{patience}")
    
    # 每5个epoch保存一次模型
    if (epoch + 1) % 5 == 0:
        model_path = os.path.join('saved_models', f'pytorch_unet_v3_epoch{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, model_path)
        print(f"Model saved to: {model_path}")
    
    # 早停检查
    if early_stop_counter >= patience:
        print(f"\nEarly stopping triggered after {patience} epochs without improvement")
        break

# 保存最终模型
final_model_path = os.path.join('saved_models', 'pytorch_unet_v3_final.pth')
torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
}, final_model_path)

print()
print("=" * 70)
print("Training complete!")
print("=" * 70)
print()
print(f"Final model saved as: {final_model_path}")
print(f"Best model saved as: {best_model_path}")
print()
print("Next step: Run predict_pytorch_v3.py to test the model")
print()
