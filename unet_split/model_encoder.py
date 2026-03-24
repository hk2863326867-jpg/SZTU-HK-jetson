"""
U-Net Encoder Model
将完整U-Net模型的编码器部分分离出来
"""
import torch
import torch.nn as nn
import torch.nn.init as init

class UNetEncoder(nn.Module):
    """U-Net编码器模型"""
    def __init__(self, in_channels=3):
        super(UNetEncoder, self).__init__()
        
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
    
    def forward(self, x):
        """
        前向传播，返回瓶颈层和特征图
        返回格式：(bottleneck, features)
        features格式：[enc1, enc2, enc3, enc4]
        """
        # 编码器
        enc1 = self.enc1(x)           # 64通道，256x256
        enc2 = self.enc2(self.pool1(enc1))  # 128通道，128x128
        enc3 = self.enc3(self.pool2(enc2))  # 256通道，64x64
        enc4 = self.enc4(self.pool3(enc3))  # 512通道，32x32
        bottleneck = self.bottleneck(self.pool4(enc4))  # 1024通道，16x16
        
        features = [enc1, enc2, enc3, enc4]
        return bottleneck, features
    
    def _init_weights(self):
        """初始化权重（与Keras的he_normal一致）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He正态分布初始化，与Keras的he_normal一致
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# 测试编码器
if __name__ == "__main__":
    print("=" * 60)
    print("Testing U-Net Encoder Model")
    print("=" * 60)
    
    # 创建编码器
    encoder = UNetEncoder(in_channels=3)
    print("Encoder created successfully!")
    
    # 测试输入
    input_tensor = torch.randn(1, 3, 256, 256)
    print(f"\nInput shape: {input_tensor.shape}")
    
    # 前向传播
    features = encoder(input_tensor)
    
    # 打印各层特征图形状
    feature_names = ['enc1', 'enc2', 'enc3', 'enc4', 'bottleneck']
    for name, feature in zip(feature_names, features):
        print(f"{name} shape: {feature.shape}")
    
    # 模型参数
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal encoder parameters: {total_params:,}")
    
    print("\n" + "=" * 60)
    print("U-Net Encoder model is working correctly!")
    print("=" * 60)
