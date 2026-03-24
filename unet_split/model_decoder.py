"""
U-Net Decoder Model
将完整U-Net模型的解码器部分分离出来
"""
import torch
import torch.nn as nn
import torch.nn.init as init

class UNetDecoder(nn.Module):
    """U-Net解码器模型"""
    def __init__(self, out_channels=1):
        super(UNetDecoder, self).__init__()
        
        # 初始化权重（与Keras的he_normal一致）
        self._init_weights()
        
        # 解码器（上采样）
        # 第一层
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # 512 + 512 = 1024
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 第二层
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 256 + 256 = 512
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 第三层
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 128 + 128 = 256
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 第四层
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64 + 64 = 128
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 输出层（简化结构）
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, bottleneck, features):
        """
        前向传播，接收瓶颈层和特征图
        bottleneck: 瓶颈层特征
        features: 跳跃连接特征列表 [enc1, enc2, enc3, enc4]
        """
        enc1, enc2, enc3, enc4 = features
        
        # 解码器
        dec1 = self.up1(bottleneck)   # 上采样到512通道
        dec1 = torch.cat([dec1, enc4], dim=1)  # 跳跃连接: 512 + 512 = 1024
        dec1 = self.dec1(dec1)        # 512通道
        
        dec2 = self.up2(dec1)         # 上采样到256通道
        dec2 = torch.cat([dec2, enc3], dim=1)  # 跳跃连接: 256 + 256 = 512
        dec2 = self.dec2(dec2)        # 256通道
        
        dec3 = self.up3(dec2)         # 上采样到128通道
        dec3 = torch.cat([dec3, enc2], dim=1)  # 跳跃连接: 128 + 128 = 256
        dec3 = self.dec3(dec3)        # 128通道
        
        dec4 = self.up4(dec3)         # 上采样到64通道
        dec4 = torch.cat([dec4, enc1], dim=1)  # 跳跃连接: 64 + 64 = 128
        dec4 = self.dec4(dec4)        # 64通道
        
        # 输出
        out = self.out(dec4)          # 1通道
        out = torch.sigmoid(out)      # 添加sigmoid激活
        
        return out
    
    def _init_weights(self):
        """初始化权重（与Keras的he_normal一致）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # 使用He正态分布初始化，与Keras的he_normal一致
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 测试解码器
if __name__ == "__main__":
    print("=" * 60)
    print("Testing U-Net Decoder Model")
    print("=" * 60)
    
    # 创建解码器
    decoder = UNetDecoder(out_channels=1)
    print("Decoder created successfully!")
    
    # 创建模拟的编码器特征图
    enc1 = torch.randn(1, 64, 256, 256)   # 64通道，256x256
    enc2 = torch.randn(1, 128, 128, 128)  # 128通道，128x128
    enc3 = torch.randn(1, 256, 64, 64)    # 256通道，64x64
    enc4 = torch.randn(1, 512, 32, 32)    # 512通道，32x32
    bottleneck = torch.randn(1, 1024, 16, 16)  # 1024通道，16x16
    
    features = [enc1, enc2, enc3, enc4, bottleneck]
    print("\nEncoder feature shapes:")
    feature_names = ['enc1', 'enc2', 'enc3', 'enc4', 'bottleneck']
    for name, feature in zip(feature_names, features):
        print(f"{name} shape: {feature.shape}")
    
    # 前向传播
    output = decoder(features)
    
    print(f"\nDecoder output shape: {output.shape}")
    print(f"Output min: {output.min().item():.4f}")
    print(f"Output max: {output.max().item():.4f}")
    
    # 模型参数
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal decoder parameters: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("U-Net Decoder model is working correctly!")
    print("=" * 60)
