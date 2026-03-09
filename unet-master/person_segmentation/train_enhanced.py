"""
使用增强数据集训练 U-Net 模型
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from model_rgb import unet_rgb
from data_rgb import trainGenerator_rgb
from tensorflow.keras.callbacks import ModelCheckpoint

print("=" * 70)
print("     Training U-Net with Enhanced Dataset")
print("=" * 70)
print()

# 数据增强参数
data_gen_args = dict(
    rotation_range=0.3,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("[1/3] Setting up data generator...")
train_path = 'enhanced_dataset/train'

# 检查数据是否存在
if not os.path.exists(train_path):
    print(f"✗ Dataset not found: {train_path}")
    print("Please run: python create_enhanced_dataset.py")
    sys.exit(1)

myGene = trainGenerator_rgb(
    batch_size=4,
    train_path=train_path,
    image_folder='image',
    mask_folder='label',
    aug_dict=data_gen_args,
    save_to_dir=None
)

print("✓ Data generator ready")
print(f"  Training data: {train_path}")
print()

print("[2/3] Creating and training model...")
model = unet_rgb()

# 模型检查点
model_checkpoint = ModelCheckpoint(
    'enhanced_unet.keras',
    monitor='loss',
    verbose=1,
    save_best_only=True
)

print("Starting training...")
print("  Batch size: 4")
print("  Steps per epoch: 200")
print("  Epochs: 5")
print("  Input size: 256x256x3 (RGB)")
print()
print("This will take about 30-60 minutes, please wait...")
print()

# 训练模型
model.fit(
    myGene,
    steps_per_epoch=200,
    epochs=5,
    callbacks=[model_checkpoint]
)

print()
print("=" * 70)
print("✓ Training complete!")
print("=" * 70)
print()
print("Model saved as: enhanced_unet.keras")
print()
print("Next step: Test with your photos")
print("Run: python predict_enhanced.py")
print()
input("Press Enter to exit...")
