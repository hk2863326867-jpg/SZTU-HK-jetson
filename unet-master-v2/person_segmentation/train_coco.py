"""
使用 COCO 数据集训练 U-Net 模型
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from model_rgb import unet_rgb
from data_rgb import trainGenerator_rgb
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("=" * 70)
print("     Training U-Net with COCO Person Dataset")
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
train_path = 'coco_person'

# 检查数据是否存在
if not os.path.exists(train_path):
    print(f"✗ Dataset not found: {train_path}")
    print("Please run: python download_coco.py")
    sys.exit(1)

# 检查图片数量
images_dir = os.path.join(train_path, 'images')
masks_dir = os.path.join(train_path, 'masks')

if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
    print("✗ Images or masks directory not found!")
    print("Please run: python download_coco.py")
    sys.exit(1)

num_images = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
num_masks = len([f for f in os.listdir(masks_dir) if f.endswith('.png')])

print(f"  Found {num_images} images")
print(f"  Found {num_masks} masks")

if num_images == 0 or num_masks == 0:
    print("✗ No training data found!")
    print("Please run: python download_coco.py")
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
    'coco_unet.keras',
    monitor='loss',
    verbose=1,
    save_best_only=True
)

# 早停（如果损失不再下降）
early_stopping = EarlyStopping(
    monitor='loss',
    patience=3,
    verbose=1
)

# 计算训练步数
steps_per_epoch = min(num_images // 8, 100)  # 每个 epoch 最多 100 步

print("Starting training...")
print(f"  Batch size: 8")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Epochs: 10")
print(f"  Input size: 256x256x3 (RGB)")
print()
print("This will take about 1-2 hours, please wait...")
print()

# 训练模型
model.fit(
    myGene,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    callbacks=[model_checkpoint, early_stopping]
)

print()
print("=" * 70)
print("✓ Training complete!")
print("=" * 70)
print()
print("Model saved as: coco_unet.keras")
print()
print("Next step: Test with your photos")
print("Run: python predict_coco.py")
print()
input("Press Enter to exit...")
