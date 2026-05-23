"""
测试CelebAMask-HQ掩码合并效果的可视化脚本
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
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

def visualize_masks(img, masks_dict, merged_mask, img_idx):
    """可视化掩码合并效果"""
    num_masks = len(masks_dict)
    fig_height = 5
    fig_width = 15
    
    if num_masks > 0:
        # 计算布局
        rows = 2
        cols = max(num_masks, 3)
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        # 显示原始图片
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # 显示各个部位掩码
        for i, (part_name, mask) in enumerate(masks_dict.items()):
            row = 0 if i < cols - 1 else 1
            col = i + 1 if i < cols - 1 else i + 1 - cols
            axes[row, col].imshow(mask, cmap='gray')
            axes[row, col].set_title(f'{part_name}')
            axes[row, col].axis('off')
        
        # 显示合并后的掩码
        axes[1, 0].imshow(merged_mask, cmap='gray')
        axes[1, 0].set_title('Merged Mask')
        axes[1, 0].axis('off')
        
        # 隐藏多余的子图
        for i in range(len(masks_dict) + 1, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(merged_mask, cmap='gray')
        axes[1].set_title('Merged Mask')
        axes[1].axis('off')
    
    plt.suptitle(f'CelebAMask-HQ Mask Visualization - Image {img_idx}')
    plt.tight_layout()
    
    # 保存图片
    output_dir = 'mask_visualization'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'mask_visualization_{img_idx}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    # 设置数据集路径
    celeb_root = '../CelebAMask-HQ'
    img_dir = os.path.join(celeb_root, 'CelebA-HQ-img')
    mask_dir = os.path.join(celeb_root, 'CelebAMask-HQ-mask-anno')
    
    # 测试图片索引
    img_idx = 0
    
    print(f"Testing mask merging for image {img_idx}")
    print("=" * 60)
    
    try:
        img, masks_dict, merged_mask = load_celebamask_data(img_idx, img_dir, mask_dir)
        visualize_masks(img, masks_dict, merged_mask, img_idx)
    except Exception as e:
        print(f"Error: {e}")
