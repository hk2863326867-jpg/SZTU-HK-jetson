"""
调试数据生成器
"""
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 数据路径
train_path = '../person_segmentation/pascal_voc_person'
image_folder = 'images'
mask_folder = 'masks'

# 数据增强参数
data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 创建数据生成器
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

print("=" * 70)
print("     Data Generator Debug")
print("=" * 70)
print()

# 检查目录结构
print("[1/4] Checking directory structure...")
print(f"Train path: {train_path}")
print(f"Image folder: {os.path.join(train_path, image_folder)}")
print(f"Mask folder: {os.path.join(train_path, mask_folder)}")

# 检查文件数量
image_files = os.listdir(os.path.join(train_path, image_folder))
mask_files = os.listdir(os.path.join(train_path, mask_folder))
print(f"Images found: {len(image_files)}")
print(f"Masks found: {len(mask_files)}")
print()

# 创建生成器
print("[2/4] Creating generators...")
image_generator = image_datagen.flow_from_directory(
    train_path,
    classes=[image_folder],
    class_mode=None,
    color_mode='rgb',
    target_size=(256, 256),
    batch_size=1,
    seed=1
)

mask_generator = mask_datagen.flow_from_directory(
    train_path,
    classes=[mask_folder],
    class_mode=None,
    color_mode='grayscale',
    target_size=(256, 256),
    batch_size=1,
    seed=1
)

# 获取一个批次的数据
print("\n[3/4] Getting batch data...")
img = next(image_generator)
mask = next(mask_generator)

print(f"Image shape: {img.shape}")
print(f"Mask shape: {mask.shape}")
print(f"Image min: {img.min()}, max: {img.max()}")
print(f"Mask min: {mask.min()}, max: {mask.max()}")

# 检查掩码是否全黑
mask_sum = mask.sum()
print(f"Mask sum: {mask_sum}")
if mask_sum == 0:
    print("WARNING: Mask is all black!")
else:
    print("Mask has content.")

# 检查图像是否正常
print(f"Image mean: {img.mean()}")
if img.mean() == 0:
    print("WARNING: Image is all black!")
else:
    print("Image has content.")

print("\n[4/4] Data validation complete!")
