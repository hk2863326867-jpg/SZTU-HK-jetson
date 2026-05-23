"""
权重提取脚本
从完整U-Net模型中提取权重并保存到编码器和解码器
"""
import os
import sys
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.model_pytorch_v3 import UNetV3
from model_encoder import UNetEncoder
from model_decoder import UNetDecoder

print("=" * 70)
print("     U-Net Weight Extraction")
print("=" * 70)
print()

# 设置模型路径（使用最佳模型）
full_model_path = '../pytorch_unet/saved_models/pytorch_unet_v3_best.pth'
encoder_save_path = 'saved_weights_best/encoder_weights_best.pth'
decoder_save_path = 'saved_weights_best/decoder_weights_best.pth'

# 创建保存目录
os.makedirs('saved_weights_best', exist_ok=True)

print("[1/3] Loading full U-Net model...")

# 加载完整模型
if not os.path.exists(full_model_path):
    print(f"  Error: Full model not found at {full_model_path}")
    sys.exit(1)

full_model = UNetV3(in_channels=3, out_channels=1)
checkpoint = torch.load(full_model_path, map_location='cpu')
full_model.load_state_dict(checkpoint['model_state_dict'])
print(f"  Full model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
print("  Full model loaded successfully!")

print("\n[2/3] Creating encoder and decoder models...")

# 创建编码器和解码器
encoder = UNetEncoder(in_channels=3)
decoder = UNetDecoder(out_channels=1)

print("  Encoder created successfully!")
print("  Decoder created successfully!")

print("\n[3/3] Extracting weights...")

# 提取编码器权重
print("  Extracting encoder weights...")
encoder_state_dict = {}

# 复制编码器相关层的权重
for name, param in full_model.named_parameters():
    if name.startswith('enc') or name.startswith('pool') or name.startswith('bottleneck'):
        encoder_state_dict[name] = param.data.clone()

encoder.load_state_dict(encoder_state_dict)
print("  Encoder weights extracted")

# 提取解码器权重
print("  Extracting decoder weights...")
decoder_state_dict = {}

# 复制解码器相关层的权重
for name, param in full_model.named_parameters():
    if name.startswith('up') or name.startswith('dec') or name.startswith('out'):
        decoder_state_dict[name] = param.data.clone()

decoder.load_state_dict(decoder_state_dict)
print("  Decoder weights extracted")

# 保存权重
print("\n[4/4] Saving weights...")
torch.save(encoder.state_dict(), encoder_save_path)
torch.save(decoder.state_dict(), decoder_save_path)

print(f"  Encoder weights saved to: {encoder_save_path}")
print(f"  Decoder weights saved to: {decoder_save_path}")

# 验证权重
print("\n[5/5] Verifying weights...")
print("  Checking encoder parameter count...")
encoder_params = sum(p.numel() for p in encoder.parameters())
print(f"    Encoder parameters: {encoder_params:,}")

print("  Checking decoder parameter count...")
decoder_params = sum(p.numel() for p in decoder.parameters())
print(f"    Decoder parameters: {decoder_params:,}")

print("  Checking full model parameter count...")
full_params = sum(p.numel() for p in full_model.parameters())
print(f"    Full model parameters: {full_params:,}")

print("  Checking parameter consistency...")
if encoder_params + decoder_params == full_params:
    print("    Parameter counts match!")
else:
    print("    Parameter counts don't match!")
    print(f"      Expected: {full_params:,}, Got: {encoder_params + decoder_params:,}")

print()
print("=" * 70)
print("Weight extraction complete!")
print("=" * 70)
print()
print("You can now use the split encoder and decoder models!")
print()
