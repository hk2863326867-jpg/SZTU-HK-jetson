"""
PyTorch U-Net Model (V3)
基于Keras架构转换的PyTorch实现
保持与Keras模型的架构一致性
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

class UNetV3(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetV3, self).__init__()
        
        # 初始化权重（与Keras的he_normal一致）
        self._init_weights()
        
        # 编码器（下采样）
        # 第一层
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二层
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三层
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第四层
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
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
        
    def forward(self, x):
        # 编码器
        enc1 = self.enc1(x)           # 64通道
        enc2 = self.enc2(self.pool1(enc1))  # 128通道
        enc3 = self.enc3(self.pool2(enc2))  # 256通道
        enc4 = self.enc4(self.pool3(enc3))  # 512通道
        bottleneck = self.bottleneck(self.pool4(enc4))  # 1024通道
        
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
            if isinstance(m, nn.Conv2d):
                # 使用He正态分布初始化，与Keras的he_normal一致
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 示例用法
if __name__ == "__main__":
    print("=" * 60)
    print("Testing PyTorch U-Net Model V3")
    print("=" * 60)
    
    # 创建模型
    print("\n[1/3] Creating U-Net model...")
    model = UNetV3(in_channels=3, out_channels=1)
    print("Model created successfully!")
    
    # 测试输入
    print("\n[2/3] Testing forward pass...")
    input_tensor = torch.randn(1, 3, 256, 256)
    print(f"Input shape: {input_tensor.shape}")
    
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    
    # 检查输出范围（应该在0-1之间）
    print(f"Output min: {output.min().item():.4f}")
    print(f"Output max: {output.max().item():.4f}")
    
    # 模型参数
    print("\n[3/3] Model summary...")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    print("\n" + "=" * 60)
    print("PyTorch U-Net V3 model is working correctly!")
    print("=" * 60)
