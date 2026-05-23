"""
验证CelebAMask-HQ掩码合并效果的脚本
不依赖matplotlib，直接保存图片验证
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import skimage.io as io
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
    
    masks_dict = {}
    if mask_files:
        print(f"Found {len(mask_files)} mask parts for image {img_idx}:")
        for mask_file in mask_files:
            part_name = os.path.basename(mask_file).split('_')[1].replace('.png', '')
            mask = io.imread(mask_file, as_gray=True)
            mask_parts.append(mask)
            masks_dict[part_name] = mask
            print(f"  - {part_name}")
        
        # 合并所有部位为一个掩码
        merged_mask = np.max(np.stack(mask_parts), axis=0)
    else:
        print(f"No masks found for image {img_idx}")
        merged_mask = np.zeros_like(img[:, :, 0])
    
    return img, masks_dict, merged_mask

def save_mask_visualization(img, masks_dict, merged_mask, img_idx):
    """保存掩码可视化结果"""
    output_dir = 'mask_verification'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始图片
    io.imsave(os.path.join(output_dir, f'{img_idx:05d}_original.jpg'), img)
    print(f"Saved original image: {img_idx:05d}_original.jpg")
    
    # 保存各个部位掩码
    for part_name, mask in masks_dict.items():
        mask_uint8 = (mask * 255).astype(np.uint8)
        io.imsave(os.path.join(output_dir, f'{img_idx:05d}_{part_name}_mask.png'), mask_uint8)
        print(f"Saved {part_name} mask: {img_idx:05d}_{part_name}_mask.png")
    
    # 保存合并后的掩码
    merged_uint8 = (merged_mask * 255).astype(np.uint8)
    io.imsave(os.path.join(output_dir, f'{img_idx:05d}_merged_mask.png'), merged_uint8)
    print(f"Saved merged mask: {img_idx:05d}_merged_mask.png")
    
    # 统计信息
    print(f"\nMask statistics:")
    print(f"  Merged mask min: {merged_mask.min():.4f}")
    print(f"  Merged mask max: {merged_mask.max():.4f}")
    print(f"  Merged mask mean: {merged_mask.mean():.4f}")
    print(f"  Non-zero pixels: {np.sum(merged_mask > 0)}")
    
    return output_dir

if __name__ == "__main__":
    # 设置数据集路径
    celeb_root = '../CelebAMask-HQ'
    img_dir = os.path.join(celeb_root, 'CelebA-HQ-img')
    mask_dir = os.path.join(celeb_root, 'CelebAMask-HQ-mask-anno')
    
    # 测试图片索引
    img_idx = 0
    
    print(f"Verifying mask merging for image {img_idx}")
    print("=" * 60)
    
    try:
        img, masks_dict, merged_mask = load_celebamask_data(img_idx, img_dir, mask_dir)
        output_dir = save_mask_visualization(img, masks_dict, merged_mask, img_idx)
        print(f"\nVerification complete! All files saved to: {output_dir}/")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
