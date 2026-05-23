"""
PyTorch U-Net V3 CelebAMask-HQ 预测脚本
使用训练好的PyTorch模型预测CelebAMask-HQ数据集
"""
import os
import sys
import numpy as np
import torch
import skimage.io as io
import skimage.transform as trans

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_pytorch_v3 import UNetV3

print("=" * 70)
print("     PyTorch U-Net V3 CelebAMask-HQ Prediction")
print("=" * 70)
print()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def testGenerator(test_path, start_idx=0, end_idx=10, target_size=(256, 256)):
    """测试数据生成器 - CelebAMask-HQ数据集"""
    for i in range(start_idx, end_idx):
        img_path = os.path.join(test_path, f"{i}.jpg")
        if not os.path.exists(img_path):
            print(f"Image {i} not found, skipping...")
            continue
        
        img = io.imread(img_path)
        
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        elif img.shape[2] == 2:
            img = img[:, :, :1]
        
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).float().unsqueeze(0)  # 添加批次维度
        
        yield img

def saveResult(save_path, npyfile, start_idx=0, threshold=0.5):
    """保存预测结果"""
    os.makedirs(os.path.join(save_path, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "thresholded"), exist_ok=True)
    
    for i, img in enumerate(npyfile):
        # 转换为numpy数组
        img = img.cpu().numpy().squeeze()
        
        # 保存原始概率图
        raw_img = (img * 255).astype(np.uint8)
        io.imsave(os.path.join(save_path, "raw", "%05d_raw.png" % (start_idx + i)), raw_img)
        
        # 应用阈值处理
        thresholded = (img > threshold).astype(np.float32)
        thresholded_img = (thresholded * 255).astype(np.uint8)
        io.imsave(os.path.join(save_path, "thresholded", "%05d_thresholded.png" % (start_idx + i)), thresholded_img)
        
        print(f"Image {start_idx + i}: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")

# 设置模型路径和测试数据路径
model_path = 'saved_models/pytorch_unet_v3_epoch50.pth'
test_path = '../CelebAMask-HQ/CelebA-HQ-img'
output_dir = 'results_celeba_pytorch_v3'

print("[1/3] Loading model...")

# 加载模型
model = UNetV3(in_channels=3, out_channels=1).to(device)

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print("  Model loaded successfully!")
else:
    print(f"  Model not found: {model_path}")
    print("  Please train the model first!")
    sys.exit(1)

model.eval()

print("\n[2/3] Setting up test data...")

if not os.path.exists(test_path):
    print(f"Test data not found: {test_path}")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

print("\n[3/3] Making predictions...")

# 生成预测结果
test_gen = testGenerator(test_path, start_idx=0, end_idx=10)
results = []

with torch.no_grad():
    for img in test_gen:
        img = img.to(device)
        output = model(img)
        results.append(output)

# 保存结果
print("\nSaving results...")
saveResult(output_dir, results, start_idx=0, threshold=0.5)

print()
print("=" * 70)
print("Prediction complete!")
print("=" * 70)
print()
print(f"Results saved to: {output_dir}")
print()
