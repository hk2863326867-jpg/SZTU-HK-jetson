"""
测试拆分后的U-Net模型
验证编码器和解码器是否能正常工作
"""
import os
import sys
import numpy as np
import torch
import skimage.io as io
import skimage.transform as trans

from model_encoder import UNetEncoder
from model_decoder import UNetDecoder

print("=" * 70)
print("     Testing Split U-Net Model")
print("=" * 70)
print()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 设置模型路径
encoder_weight_path = 'saved_weights/encoder_weights.pth'
decoder_weight_path = 'saved_weights/decoder_weights.pth'

# 设置测试图片路径
test_img_path = '../CelebAMask-HQ/CelebA-HQ-img/0.jpg'
output_dir = 'test_results'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

print("[1/4] Loading encoder and decoder models...")

# 创建编码器和解码器
encoder = UNetEncoder(in_channels=3).to(device)
decoder = UNetDecoder(out_channels=1).to(device)

# 加载权重
if not os.path.exists(encoder_weight_path):
    print(f"  Error: Encoder weights not found at {encoder_weight_path}")
    print("  Please run extract_weights.py first!")
    sys.exit(1)

if not os.path.exists(decoder_weight_path):
    print(f"  Error: Decoder weights not found at {decoder_weight_path}")
    print("  Please run extract_weights.py first!")
    sys.exit(1)

encoder.load_state_dict(torch.load(encoder_weight_path, map_location=device))
decoder.load_state_dict(torch.load(decoder_weight_path, map_location=device))

print("  ✓ Encoder weights loaded")
print("  ✓ Decoder weights loaded")

# 设置模型为评估模式
encoder.eval()
decoder.eval()

print("\n[2/4] Loading test image...")

# 加载测试图片
if not os.path.exists(test_img_path):
    print(f"  Error: Test image not found at {test_img_path}")
    sys.exit(1)

img = io.imread(test_img_path)

# 处理图像通道
if len(img.shape) == 2:
    img = np.stack([img, img, img], axis=2)
elif img.shape[2] == 4:
    img = img[:, :, :3]
elif img.shape[2] == 2:
    img = img[:, :, :1]

# 保存原始图片
io.imsave(os.path.join(output_dir, 'original.jpg'), img)

print(f"  Original image shape: {img.shape}")
original_height, original_width = img.shape[0], img.shape[1]

# 预处理
img_processed = img / 255
img_processed = trans.resize(img_processed, (256, 256))
img_processed = np.transpose(img_processed, (2, 0, 1))  # (H, W, C) -> (C, H, W)
img_tensor = torch.from_numpy(img_processed).float().unsqueeze(0).to(device)  # 添加批次维度

print(f"  Preprocessed image shape: {img_tensor.shape}")

print("\n[3/4] Running inference...")

with torch.no_grad():
    # 编码器前向传播
    bottleneck, features = encoder(img_tensor)
    print(f"  Encoder output:")
    print(f"    Bottleneck: {bottleneck.shape}")
    feature_names = ['enc1', 'enc2', 'enc3', 'enc4']
    for name, feature in zip(feature_names, features):
        print(f"    {name}: {feature.shape}")
    
    # 解码器前向传播
    output = decoder(bottleneck, features)
    print(f"\n  Decoder output shape: {output.shape}")
    print(f"  Output min: {output.min().item():.4f}")
    print(f"  Output max: {output.max().item():.4f}")
    print(f"  Output mean: {output.mean().item():.4f}")

print("\n[4/4] Saving results...")

# 保存结果
output_np = torch.sigmoid(output).cpu().numpy()[0, 0]

# 将输出调整回原始尺寸
original_height, original_width = img.shape[0], img.shape[1]
raw_resized = trans.resize(output_np, (original_height, original_width))
raw_img = (raw_resized * 255).astype(np.uint8)
io.imsave(os.path.join(output_dir, 'raw_prediction.png'), raw_img)

# 使用更高阈值获得更清晰的轮廓
threshold = 0.7
thresholded = (output_np > threshold).astype(np.float32)

# 形态学操作清理边缘
from scipy import ndimage
# 腐蚀去除小噪点
thresholded = ndimage.binary_erosion(thresholded, structure=np.ones((3, 3)))
# 膨胀恢复主体
thresholded = ndimage.binary_dilation(thresholded, structure=np.ones((5, 5)))

# 将阈值处理后的输出调整回原始尺寸
thresholded_resized = trans.resize(thresholded, (original_height, original_width))
thresholded_img = (thresholded_resized * 255).astype(np.uint8)
io.imsave(os.path.join(output_dir, 'thresholded_prediction.png'), thresholded_img)

print("  ✓ Raw prediction saved")
print("  ✓ Thresholded prediction saved")

# 统计信息
print("\n[5/5] Results summary...")
print(f"  Raw prediction - min: {output_np.min():.4f}, max: {output_np.max():.4f}, mean: {output_np.mean():.4f}")
print(f"  Thresholded prediction - foreground ratio: {thresholded.mean():.4f}")

print()
print("=" * 70)
print("Test completed successfully!")
print("=" * 70)
print()
print(f"Results saved to: {output_dir}")
print()
