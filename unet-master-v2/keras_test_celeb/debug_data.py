"""
调试数据生成器的脚本
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob

def load_celebamask_data(img_idx, img_dir, mask_dir):
    """加载CelebAMask-HQ数据"""
    # 加载图片
    img_path = os.path.join(img_dir, f"{img_idx}.jpg")
    img = io.imread(img_path)
    
    # 加载所有部位的掩码并合并
    mask_parts = []
    subfolder = str(img_idx // 2000)  # 每2000张图片一个文件夹
    mask_subdir = os.path.join(mask_dir, subfolder)
    
    # 查找所有掩码文件
    mask_files = glob.glob(os.path.join(mask_subdir, f"{img_idx:05d}_*.png"))
    
    if mask_files:
        print(f"Found {len(mask_files)} mask parts for image {img_idx}")
        for mask_file in mask_files:
            mask = io.imread(mask_file, as_gray=True)
            mask_parts.append(mask)
        
        # 合并所有部位为一个掩码
        mask = np.max(np.stack(mask_parts), axis=0)
    else:
        print(f"No masks found for image {img_idx}")
        mask = np.zeros_like(img[:, :, 0])
    
    return img, mask

def trainGenerator(batch_size, img_dir, mask_dir, aug_dict, num_images=100):
    """训练数据生成器"""
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    batch_images = []
    batch_masks = []
    
    while True:
        for img_idx in range(num_images):
            try:
                img, mask = load_celebamask_data(img_idx, img_dir, mask_dir)
                
                # 预处理
                img = img / 255.0
                mask = mask / 255.0
                
                # 调整大小
                img = trans.resize(img, (256, 256))
                mask = trans.resize(mask, (256, 256))
                
                # 扩展维度
                mask = np.expand_dims(mask, axis=-1)
                
                # 数据增强
                img_aug = image_datagen.random_transform(img)
                mask_aug = mask_datagen.random_transform(mask)
                
                batch_images.append(img_aug)
                batch_masks.append(mask_aug)
                
                # 当批次满了就yield
                if len(batch_images) == batch_size:
                    yield (np.array(batch_images), np.array(batch_masks))
                    batch_images = []
                    batch_masks = []
                    
            except Exception as e:
                print(f"Error loading image {img_idx}: {e}")
                continue
        
        # 如果最后一批不满batch_size，也yield
        if batch_images:
            yield (np.array(batch_images), np.array(batch_masks))
            batch_images = []
            batch_masks = []

# 数据增强配置
data_gen_args = dict(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 设置数据集路径
celeb_root = '../CelebAMask-HQ'
img_dir = os.path.join(celeb_root, 'CelebA-HQ-img')
mask_dir = os.path.join(celeb_root, 'CelebAMask-HQ-mask-anno')

# 创建数据生成器
myGene = trainGenerator(
    batch_size=2,
    img_dir=img_dir,
    mask_dir=mask_dir,
    aug_dict=data_gen_args,
    num_images=5
)

# 获取一批数据并检查
print("Testing data generator...")
print("=" * 60)

try:
    X, y = next(myGene)
    print(f"Batch size: {X.shape[0]}")
    print(f"Image shape: {X.shape}")
    print(f"Mask shape: {y.shape}")
    
    # 检查第一张图片
    print("\nFirst image statistics:")
    print(f"  Min: {X[0].min():.4f}, Max: {X[0].max():.4f}, Mean: {X[0].mean():.4f}")
    
    # 检查第一张掩码
    print("\nFirst mask statistics:")
    print(f"  Min: {y[0].min():.4f}, Max: {y[0].max():.4f}, Mean: {y[0].mean():.4f}")
    print(f"  Non-zero pixels: {np.sum(y[0] > 0)}")
    print(f"  Total pixels: {y[0].shape[0] * y[0].shape[1]}")
    print(f"  Percentage of mask: {np.sum(y[0] > 0) / (y[0].shape[0] * y[0].shape[1]):.4f}")
    
    # 保存示例图片和掩码
    output_dir = 'debug_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存第一张图片
    img_uint8 = (X[0] * 255).astype(np.uint8)
    io.imsave(os.path.join(output_dir, 'example_image.jpg'), img_uint8)
    print(f"\nExample image saved to: {output_dir}/example_image.jpg")
    
    # 保存第一张掩码
    mask_uint8 = (y[0, :, :, 0] * 255).astype(np.uint8)
    io.imsave(os.path.join(output_dir, 'example_mask.png'), mask_uint8)
    print(f"Example mask saved to: {output_dir}/example_mask.png")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
