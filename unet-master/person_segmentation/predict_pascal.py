"""
使用 Pascal VOC 训练的 U-Net 模型进行预测
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skimage.io as io
import skimage.transform as trans
from model_rgb import unet_rgb

print("=" * 70)
print("     Pascal VOC U-Net Prediction")
print("=" * 70)
print()

# 加载模型
print("[1/2] Loading Pascal VOC-trained model...")

if not os.path.exists('pascal_unet.keras'):
    print("✗ Model not found: pascal_unet.keras")
    print("Please run: python train_pascal.py")
    sys.exit(1)

model = unet_rgb()
model.load_weights('pascal_unet.keras')
print("✓ Model loaded from pascal_unet.keras")
print()

# 预测
print("[2/2] Making predictions...")
test_path = "dataset/test"

if not os.path.exists(test_path):
    print(f"✗ Test directory not found: {test_path}")
    sys.exit(1)

# 获取测试图片
image_files = [f for f in os.listdir(test_path) 
               if f.endswith(('.png', '.jpg', '.jpeg')) 
               and '_predict' not in f 
               and '_pretrained' not in f
               and '_enhanced' not in f
               and '_coco' not in f
               and '_real' not in f
               and '_pascal' not in f]

if len(image_files) == 0:
    print("✗ No test images found!")
    sys.exit(1)

print(f"Found {len(image_files)} test images")
print()

results = []
for i, img_file in enumerate(sorted(image_files)):
    img_path = os.path.join(test_path, img_file)
    
    # 读取图片
    img = io.imread(img_path)
    
    # 处理不同格式
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    
    # 预处理
    img = img / 255.0
    img = trans.resize(img, (256, 256))
    img = np.reshape(img, (1,) + img.shape)
    
    # 预测
    pred = model.predict(img, verbose=0)
    results.append((img_file, pred[0]))
    
    print(f"  Predicted {i+1}/{len(image_files)}: {img_file}")

# 保存结果
print()
print("Saving results...")
for img_file, pred in results:
    # 生成输出文件名
    base_name = os.path.splitext(img_file)[0]
    output_name = f"{base_name}_pascal.png"
    output_path = os.path.join(test_path, output_name)
    
    # 处理预测结果
    img = pred[:, :, 0]
    img = (img * 255).astype(np.uint8)
    
    # 保存
    io.imsave(output_path, img)
    print(f"  Saved: {output_name}")

print()
print("=" * 70)
print("✓ All predictions complete!")
print("=" * 70)
print()
print("Results saved with '_pascal.png' suffix")
print()
print("Compare all results:")
print("  - Original U-Net (10 stick figures): *_predict.png")
print("  - Color clustering method: *_pretrained.png")
print("  - Enhanced U-Net (50 better stick figures): *_enhanced.png")
print("  - Real-style U-Net (diverse person data): *_real.png")
print("  - Pascal VOC U-Net (real photos): *_pascal.png")
print()
input("Press Enter to exit...")
