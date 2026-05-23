"""
使用真实风格数据集训练 U-Net 模型
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from model_rgb import unet_rgb
from data_rgb import trainGenerator_rgb
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("=" * 70)
print("     Training U-Net with Real-Style Dataset")
print("=" * 70)
print()

# 数据增强参数
data_gen_args = dict(
    rotation_range=0.25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("[1/3] Setting up data generator...")
train_path = 'real_person_dataset'

# 检查数据是否存在
if not os.path.exists(train_path):
    print(f"✗ Dataset not found: {train_path}")
    print("Please run: python download_real_images.py")
    sys.exit(1)

images_dir = os.path.join(train_path, 'images')
masks_dir = os.path.join(train_path, 'masks')

if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
    print("✗ Images or masks directory not found!")
    print("Please run: python download_real_images.py")
    sys.exit(1)

# 计算训练图片数量（排除测试图片）
train_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg') and not f.startswith('test_')]
num_images = len(train_images)

print(f"  Found {num_images} training images")

if num_images == 0:
    print("✗ No training data found!")
    sys.exit(1)

myGene = trainGenerator_rgb(
    batch_size=8,
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
    'real_unet.keras',
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

# 计算训练步数
steps_per_epoch = min(num_images // 8, 100)
num_epochs = min(20, max(10, num_images // 50))  # 根据数据量调整 epoch

print("Starting training...")
print(f"  Batch size: 8")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Epochs: {num_epochs}")
print(f"  Input size: 256x256x3 (RGB)")
print()
print("This will take about 30-60 minutes, please wait...")
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
print("Model saved as: real_unet.keras")
print()
print("Next step: Test with your photos")
print("Run: python predict_real.py")
print()
input("Press Enter to exit...")
