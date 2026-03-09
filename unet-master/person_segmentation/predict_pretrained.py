"""
使用简单方法进行人物分割预测
使用纯 PIL 和 numpy，无其他依赖
"""
import os
import sys
import numpy as np

print("=" * 70)
print("     Person Segmentation - Simple Background Removal")
print("=" * 70)
print()

test_path = "dataset/test"

# 检查测试图片
if not os.path.exists(test_path):
    print(f"✗ Test directory not found: {test_path}")
    sys.exit(1)

image_files = [f for f in os.listdir(test_path) 
               if f.endswith(('.png', '.jpg', '.jpeg')) and '_predict' not in f and not f.endswith('_pretrained.png')]

if len(image_files) == 0:
    print("✗ No test images found!")
    sys.exit(1)

print(f"Found {len(image_files)} test images")
print()

# 尝试导入 PIL
try:
    from PIL import Image
    print("[Method] Using PIL for simple background removal")
except ImportError:
    print("✗ PIL not available!")
    print("Please install: pip install Pillow")
    sys.exit(1)

def simple_dilation(mask, iterations=1):
    """简单的膨胀操作"""
    result = mask.copy()
    for _ in range(iterations):
        # 使用卷积实现膨胀
        padded = np.pad(result, 1, mode='constant', constant_values=0)
        result = np.maximum.reduce([
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
            padded[1:-1, 1:-1]
        ])
    return result

def simple_erosion(mask, iterations=1):
    """简单的腐蚀操作"""
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode='constant', constant_values=1)
        result = np.minimum.reduce([
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
            padded[1:-1, 1:-1]
        ])
    return result

def simple_background_removal(image_path, output_path):
    """
    使用简单的图像处理方法进行前景分割
    基于颜色聚类
    """
    try:
        # 打开图像
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img, dtype=np.float32)
        
        # 获取图像尺寸
        h, w = img_array.shape[:2]
        
        # 方法：基于颜色差异的简单分割
        # 假设人物在图像中心，背景在边缘
        
        # 计算图像中心区域的颜色均值（假设是人物）
        center_y, center_x = h // 2, w // 2
        margin_y, margin_x = h // 4, w // 4
        
        # 提取中心区域
        y1 = max(0, center_y - margin_y)
        y2 = min(h, center_y + margin_y)
        x1 = max(0, center_x - margin_x)
        x2 = min(w, center_x + margin_x)
        
        center_region = img_array[y1:y2, x1:x2]
        
        # 计算中心区域的颜色均值
        center_mean = np.mean(center_region, axis=(0, 1))
        
        # 创建掩码：与中心颜色相似的区域为前景
        # 计算每个像素与中心颜色的距离
        diff = np.abs(img_array - center_mean)
        distance = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # 阈值：距离小于一定值为前景
        threshold = np.mean(distance) * 0.7
        mask = (distance < threshold).astype(np.uint8) * 255
        
        # 简单的形态学操作：膨胀然后腐蚀，填充小空洞
        mask_bool = mask > 0
        mask_bool = simple_dilation(mask_bool, iterations=3)
        mask_bool = simple_erosion(mask_bool, iterations=2)
        mask = mask_bool.astype(np.uint8) * 255
        
        # 创建透明背景图像
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = img_array.astype(np.uint8)
        rgba[:, :, 3] = mask
        
        # 保存
        output_img = Image.fromarray(rgba, 'RGBA')
        output_img.save(output_path)
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def fallback_segmentation(image_path, output_path):
    """
    备选方法：使用简单的阈值分割
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        
        # 转换为灰度
        gray = np.mean(img_array, axis=2)
        
        # 简单的阈值分割
        threshold = np.mean(gray)
        mask = (gray < threshold + 30).astype(np.uint8) * 255
        
        # 创建透明背景图像
        h, w = img_array.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = img_array
        rgba[:, :, 3] = mask
        
        output_img = Image.fromarray(rgba, 'RGBA')
        output_img.save(output_path)
        return True
        
    except Exception as e:
        print(f"  ✗ Fallback also failed: {e}")
        return False

# 处理所有图片
success_count = 0
for i, img_file in enumerate(sorted(image_files)):
    img_path = os.path.join(test_path, img_file)
    output_name = os.path.splitext(img_file)[0] + "_pretrained.png"
    output_path = os.path.join(test_path, output_name)
    
    print(f"  Processing {img_file}...")
    
    # 尝试主要方法
    if simple_background_removal(img_path, output_path):
        print(f"    ✓ Saved: {output_name}")
        success_count += 1
    else:
        # 尝试备选方法
        print(f"    Trying fallback method...")
        if fallback_segmentation(img_path, output_path):
            print(f"    ✓ Saved: {output_name} (fallback)")
            success_count += 1
        else:
            print(f"    ✗ Failed completely")

print()
print("=" * 70)
print(f"✓ Processed {success_count}/{len(image_files)} images")
print("=" * 70)
print()
print("Results saved with '_pretrained.png' suffix")
print()
print("Note: This is a simple method based on color clustering.")
print("Results may vary depending on image content.")
print()
print("For professional results, consider using:")
print("  - rembg library: pip install rembg")
print("  - Deep learning models like U-Net with proper training")
print()
input("Press Enter to exit...")
