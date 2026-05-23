"""
使用 Pascal VOC 数据集训练 U-Net 模型
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from model_rgb import unet_rgb
from data_rgb import trainGenerator_rgb
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("=" * 70)
print("     Training U-Net with Pascal VOC Person Dataset")
print("=" * 70)
print()

# 数据增强参数
data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("[1/3] Setting up data generator...")
train_path = 'pascal_voc_person'

# 检查数据是否存在
if not os.path.exists(train_path):
    print(f"✗ Dataset not found: {train_path}")
    print("Please run: python download_pascal_voc.py")
    sys.exit(1)

images_dir = os.path.join(train_path, 'images')
masks_dir = os.path.join(train_path, 'masks')

if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
    print("✗ Images or masks directory not found!")
    print("Please run: python download_pascal_voc.py")
    sys.exit(1)

# 计算训练图片数量
train_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
num_images = len(train_images)

print(f"  Found {num_images} training images")

if num_images == 0:
    print("✗ No training data found!")
    print("Please run: python download_pascal_voc.py")
    sys.exit(1)

# 限制训练图片数量以减少内存使用
max_train_images = 50  # 只用前 50 张，进一步减少
if num_images > max_train_images:
    print(f"  Using first {max_train_images} images to save memory")
    num_images = max_train_images

myGene = trainGenerator_rgb(
    batch_size=2,  # 批次大小减为 2
    train_path=train_path,
    image_folder='images',
    mask_folder='masks',
    aug_dict=data_gen_args,
    save_to_dir=None
)

print("✓ Data generator ready")
print()

print("[2/3] Creating and training model...")
model = unet_rgb()

# 模型检查点
model_checkpoint = ModelCheckpoint(
    'pascal_unet.keras',
    monitor='loss',
    verbose=1,
    save_best_only=True
)

# 早停
early_stopping = EarlyStopping(
    monitor='loss',
    patience=5,
    verbose=1
)

# 计算训练参数
steps_per_epoch = min(num_images // 2, 25)  # 减少步数
num_epochs = 5  # 只训练 5 轮

print("Starting training...")
print(f"  Batch size: 2")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Epochs: {num_epochs}")
print(f"  Training images: {num_images}")
print(f"  Input size: 256x256x3 (RGB)")
print()
print("This will take about 15-30 minutes, please wait...")
print()

# 训练模型
model.fit(
    myGene,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    callbacks=[model_checkpoint, early_stopping]
)

print()
print("=" * 70)
print("✓ Training complete!")
print("=" * 70)
print()
print("Model saved as: pascal_unet.keras")
print()
print("Next step: Test with your photos")
print("Run: python predict_pascal.py")
print()
input("Press Enter to exit...")
