"""
创建增强版人物分割数据集
生成更多样化、更真实的合成人物图像
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import random

print("=" * 70)
print("     Creating Enhanced Person Segmentation Dataset")
print("=" * 70)
print()

# 创建目录
dataset_dir = "enhanced_dataset"
train_image_dir = os.path.join(dataset_dir, "train", "image")
train_label_dir = os.path.join(dataset_dir, "train", "label")
test_image_dir = os.path.join(dataset_dir, "test")

for d in [train_image_dir, train_label_dir, test_image_dir]:
    os.makedirs(d, exist_ok=True)

print("Creating 50 enhanced training samples...")
print()

def draw_person(draw_img, draw_mask, center_x, center_y, scale=1.0):
    """绘制一个更复杂的人物形状"""
    
    # 随机颜色
    skin_color = (random.randint(180, 220), random.randint(140, 180), random.randint(100, 140))
    shirt_color = (random.randint(50, 150), random.randint(50, 150), random.randint(100, 200))
    pants_color = (random.randint(30, 80), random.randint(30, 80), random.randint(80, 150))
    
    # 头部（椭圆）
    head_w = int(25 * scale)
    head_h = int(30 * scale)
    head_y = center_y - int(70 * scale)
    draw_img.ellipse([center_x - head_w, head_y - head_h, 
                      center_x + head_w, head_y + head_h], 
                     fill=skin_color)
    draw_mask.ellipse([center_x - head_w, head_y - head_h, 
                       center_x + head_w, head_y + head_h], 
                      fill=255)
    
    # 身体（圆角矩形）
    body_w = int(40 * scale)
    body_h = int(80 * scale)
    body_y = center_y - int(20 * scale)
    draw_img.rounded_rectangle([center_x - body_w, body_y,
                                center_x + body_w, body_y + body_h], 
                               radius=10, fill=shirt_color)
    draw_mask.rounded_rectangle([center_x - body_w, body_y,
                                 center_x + body_w, body_y + body_h], 
                                radius=10, fill=255)
    
    # 左腿
    leg_w = int(15 * scale)
    leg_h = int(70 * scale)
    leg_y = body_y + body_h
    draw_img.rectangle([center_x - body_w + 5, leg_y,
                        center_x - body_w + 5 + leg_w * 2, leg_y + leg_h], 
                       fill=pants_color)
    draw_mask.rectangle([center_x - body_w + 5, leg_y,
                         center_x - body_w + 5 + leg_w * 2, leg_y + leg_h], 
                        fill=255)
    
    # 右腿
    draw_img.rectangle([center_x + body_w - 5 - leg_w * 2, leg_y,
                        center_x + body_w - 5, leg_y + leg_h], 
                       fill=pants_color)
    draw_mask.rectangle([center_x + body_w - 5 - leg_w * 2, leg_y,
                         center_x + body_w - 5, leg_y + leg_h], 
                        fill=255)
    
    # 左臂
    arm_w = int(12 * scale)
    arm_h = int(60 * scale)
    arm_y = body_y + 10
    draw_img.rectangle([center_x - body_w - arm_w, arm_y,
                        center_x - body_w, arm_y + arm_h], 
                       fill=skin_color)
    draw_mask.rectangle([center_x - body_w - arm_w, arm_y,
                         center_x - body_w, arm_y + arm_h], 
                        fill=255)
    
    # 右臂
    draw_img.rectangle([center_x + body_w, arm_y,
                        center_x + body_w + arm_w, arm_y + arm_h], 
                       fill=skin_color)
    draw_mask.rectangle([center_x + body_w, arm_y,
                         center_x + body_w + arm_w, arm_y + arm_h], 
                        fill=255)

# 生成训练数据
num_train = 50
for i in range(num_train):
    # 随机背景颜色
    bg_color = (random.randint(80, 180), random.randint(80, 180), random.randint(80, 180))
    img = Image.new('RGB', (256, 256), bg_color)
    mask = Image.new('L', (256, 256), 0)
    
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)
    
    # 随机位置和大小
    center_x = 128 + random.randint(-30, 30)
    center_y = 128 + random.randint(-20, 20)
    scale = random.uniform(0.8, 1.2)
    
    # 绘制人物
    draw_person(draw_img, draw_mask, center_x, center_y, scale)
    
    # 添加随机噪声
    img_array = np.array(img)
    noise = np.random.normal(0, 15, img_array.shape).astype(np.int16)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # 保存
    img.save(os.path.join(train_image_dir, f'{i:04d}.jpg'))
    mask.save(os.path.join(train_label_dir, f'{i:04d}.png'))
    
    if (i + 1) % 10 == 0:
        print(f"  Created {i + 1}/{num_train} training samples")

print()
print("Creating 10 test samples...")

# 生成测试数据
num_test = 10
for i in range(num_test):
    bg_color = (random.randint(80, 180), random.randint(80, 180), random.randint(80, 180))
    img = Image.new('RGB', (256, 256), bg_color)
    mask = Image.new('L', (256, 256), 0)
    
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)
    
    center_x = 128 + random.randint(-30, 30)
    center_y = 128 + random.randint(-20, 20)
    scale = random.uniform(0.8, 1.2)
    
    draw_person(draw_img, draw_mask, center_x, center_y, scale)
    
    img_array = np.array(img)
    noise = np.random.normal(0, 15, img_array.shape).astype(np.int16)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    img.save(os.path.join(test_image_dir, f'{i:04d}.jpg'))
    mask.save(os.path.join(test_image_dir, f'{i:04d}_mask.png'))

print()
print("=" * 70)
print("✓ Enhanced dataset created!")
print("=" * 70)
print()
print(f"Training set: {num_train} images")
print(f"  Images: {train_image_dir}")
print(f"  Labels: {train_label_dir}")
print()
print(f"Test set: {num_test} images")
print(f"  Images: {test_image_dir}")
print()
print("Next step: Train U-Net with this enhanced dataset")
print("Run: python train_enhanced.py")
print()
input("Press Enter to exit...")
