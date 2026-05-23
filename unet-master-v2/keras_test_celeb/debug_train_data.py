import numpy as np
from skimage import io, transform as trans
import os
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
            # 读取为灰度图
            mask = io.imread(mask_file, as_gray=True)
            # 确保掩码是0-1范围
            if mask.max() > 1.0:
                mask = mask / 255.0
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
                # 掩码已经是0-1范围，不需要再除以255
                
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

# 测试数据生成器
if __name__ == "__main__":
    print("Testing training data generator...")
    
    # 设置路径
    img_dir = "d:/OneDrive/桌面/github/unet-master/CelebAMask-HQ/CelebA-HQ-img"
    mask_dir = "d:/OneDrive/桌面/github/unet-master/CelebAMask-HQ/CelebAMask-HQ-mask-anno"
    
    # 设置数据增强参数
    aug_dict = {
        'rotation_range': 10,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True,
        'brightness_range': [0.8, 1.2],
        'fill_mode': 'nearest'
    }
    
    # 创建数据生成器
    generator = trainGenerator(batch_size=2, img_dir=img_dir, mask_dir=mask_dir, aug_dict=aug_dict, num_images=10)
    
    # 获取一个批次的数据
    batch_images, batch_masks = next(generator)
    
    print(f"\nBatch size: {len(batch_images)}")
    print(f"Image shape: {batch_images.shape}")
    print(f"Mask shape: {batch_masks.shape}")
    
    # 打印第一张图片的统计信息
    print("\nFirst image statistics:")
    print(f"  Min: {batch_images[0].min():.4f}, Max: {batch_images[0].max():.4f}, Mean: {batch_images[0].mean():.4f}")
    
    # 打印第一张掩码的统计信息
    print("\nFirst mask statistics:")
    print(f"  Min: {batch_masks[0].min():.4f}, Max: {batch_masks[0].max():.4f}, Mean: {batch_masks[0].mean():.4f}")
    print(f"  Non-zero pixels: {np.sum(batch_masks[0] > 0.5)}")
    print(f"  Total pixels: {batch_masks[0].size}")
    print(f"  Percentage of mask: {np.sum(batch_masks[0] > 0.5) / batch_masks[0].size:.4f}")
    
    # 保存示例图片和掩码
    os.makedirs("debug_train_data", exist_ok=True)
    io.imsave("debug_train_data/example_image.jpg", (batch_images[0] * 255).astype(np.uint8))
    io.imsave("debug_train_data/example_mask.png", (batch_masks[0, :, :, 0] * 255).astype(np.uint8))
    print("\nExample image saved to: debug_train_data/example_image.jpg")
    print("Example mask saved to: debug_train_data/example_mask.png")
