import numpy as np
from skimage import io
import os

# 测试图片加载
img_path = "d:/OneDrive/桌面/github/unet-master/CelebAMask-HQ/CelebA-HQ-img/0.jpg"
print(f"Loading image from: {img_path}")

if os.path.exists(img_path):
    img = io.imread(img_path)
    print(f"Image loaded successfully!")
    print(f"Shape: {img.shape}")
    print(f"Min: {img.min()}")
    print(f"Max: {img.max()}")
    print(f"Mean: {img.mean()}")
    
    # 保存原始图片用于检查
    io.imsave("test_image.jpg", img)
    print("Original image saved as: test_image.jpg")
else:
    print("Image file does not exist!")

# 测试掩码加载
mask_path = "d:/OneDrive/桌面/github/unet-master/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_skin.png"
print(f"\nLoading mask from: {mask_path}")

if os.path.exists(mask_path):
    mask = io.imread(mask_path, as_gray=True)
    print(f"Mask loaded successfully!")
    print(f"Shape: {mask.shape}")
    print(f"Min: {mask.min()}")
    print(f"Max: {mask.max()}")
    print(f"Mean: {mask.mean()}")
    print(f"Non-zero pixels: {np.sum(mask > 0)}")
    
    # 保存掩码用于检查
    io.imsave("test_mask.png", (mask * 255).astype(np.uint8))
    print("Mask saved as: test_mask.png")
else:
    print("Mask file does not exist!")
