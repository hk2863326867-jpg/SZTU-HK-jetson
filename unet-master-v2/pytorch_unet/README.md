# PyTorch U-Net 人物图像分割

基于 PyTorch 实现的 U-Net 架构，用于人物图像分割任务。本项目使用 CelebAMask-HQ 数据集进行训练，能够实现准确的人物分割。

## 📋 目录结构

```
pytorch_unet/
├── model_pytorch_v3.py              # U-Net V3 模型定义（最新版本）
├── train_pytorch_v3.py              # CelebAMask-HQ 数据集训练脚本（优化版）
├── predict_pytorch_v3_celeba.py     # CelebAMask-HQ 预测脚本
├── saved_models/                    # 训练好的模型权重
│   ├── pytorch_unet_v3_best.pth     # 最佳模型（验证损失最低）
│   ├── pytorch_unet_v3_epoch5.pth   # 第5轮模型
│   ├── pytorch_unet_v3_epoch10.pth  # 第10轮模型
│   ├── ...
│   ├── pytorch_unet_v3_epoch50.pth  # 第50轮模型
│   └── pytorch_unet_v3_final.pth    # 最终模型（第50轮）
├── results_celeba_pytorch_v3/       # 预测结果
│   ├── raw/                         # 原始概率图
│   └── thresholded/                 # 阈值处理后的掩码图
└── README.md                        # 项目文档
```

## 🔧 环境要求

- Python 3.10
- PyTorch 2.0+
- torchvision
- numpy
- scikit-image

### 安装依赖

```bash
pip install torch torchvision numpy scikit-image
```

## 🚀 快速开始

### 1. 使用预训练模型进行预测

```bash
python predict_pytorch_v3_celeba.py
```

**说明**：
- 输入：`../CelebAMask-HQ/CelebA-HQ-img/` 文件夹中的图片（索引0-9）
- 输出：`results_celeba_pytorch_v3/` 文件夹
- 支持切换不同轮次的模型（在脚本中修改 `model_path`）

### 2. 训练自己的模型

```bash
python train_pytorch_v3.py
```

**训练配置**：
- 训练数据：2000 张 CelebAMask-HQ 图片
- 训练轮数：50 epochs
- 学习率：1e-4（带自动调整）
- 批次大小：4
- 输入尺寸：256x256
- 验证集比例：20%

**训练数据准备**：
```
CelebAMask-HQ/
├── CelebA-HQ-img/        # 原始图片（1024x1024）
└── CelebAMask-HQ-mask-anno/  # 掩码标注（按子文件夹分类）
```

## 🔧 训练优化特性

### 数据增强
- ✅ 随机水平翻转
- ✅ 随机旋转（90度倍数）
- ✅ 随机亮度/对比度调整
- ✅ 随机高斯噪声添加

### 训练策略
- ✅ 训练集/验证集划分（8:2比例）
- ✅ 学习率自动调整（ReduceLROnPlateau）
- ✅ 早停机制（10轮无改善自动停止）
- ✅ 最佳模型自动保存
- ✅ 每5轮保存检查点模型

## 📊 模型架构

### U-Net V3 结构

```
输入 (3x256x256)
  ↓
编码器（下采样）
  ├─ enc1: 64通道 (256x256)
  ├─ enc2: 128通道 (128x128)
  ├─ enc3: 256通道 (64x64)
  ├─ enc4: 512通道 (32x32)
  ↓
瓶颈层: 1024通道 (16x16)
  ↓
解码器（上采样）+ 跳跃连接
  ├─ dec1: 512通道 (32x32)  ← 连接 enc4
  ├─ dec2: 256通道 (64x64)  ← 连接 enc3
  ├─ dec3: 128通道 (128x128) ← 连接 enc2
  ├─ dec4: 64通道 (256x256)  ← 连接 enc1
  ↓
输出 (1x256x256) - Sigmoid激活
```

### 模型特点

- **输入**：RGB 图像（3 通道）
- **输出**：分割概率图（单通道，值范围 0-1）
- **参数量**：约 3100 万
- **激活函数**：ReLU（隐藏层），Sigmoid（输出层）
- **损失函数**：BCELoss（二元交叉熵）

## 📈 输出说明

### 概率图输出

模型输出的是**概率图**，不是二值图像：
- **高概率值（接近 1）**：人物区域（显示为白色/亮色）
- **低概率值（接近 0）**：背景区域（显示为黑色/暗色）
- **中间值**：过渡区域（显示为灰色）

### 预测结果示例

```
Image 0: min=0.0000, max=1.0000, mean=0.8590
Image 1: min=0.0001, max=1.0000, mean=0.8542
```

**解读**：
- 输出范围表示概率值的最小值和最大值
- 均值越高，表示图像中人物区域占比越大

## � 自定义预测

### 切换不同轮次的模型

修改 `predict_pytorch_v3_celeba.py`：

```python
# 使用最佳模型
model_path = 'saved_models/pytorch_unet_v3_best.pth'

# 使用第10轮模型
model_path = 'saved_models/pytorch_unet_v3_epoch10.pth'

# 使用最终模型
model_path = 'saved_models/pytorch_unet_v3_final.pth'
```

### 调整预测参数

```python
# 修改预测图片范围
test_gen = testGenerator(test_path, start_idx=0, end_idx=10)

# 修改阈值
threshold = 0.5  # 默认值
```

## 💡 训练技巧

### 调整超参数

修改 `train_pytorch_v3.py`：

```python
# 训练数据量
max_train_images = 2000  # 默认 2000

# 训练轮数
num_epochs = 50  # 默认 50

# 批次大小
batch_size = 4  # 默认 4

# 学习率
learning_rate = 1e-4  # 默认 1e-4

# 早停耐心值
patience = 10  # 默认 10
```

### 性能优化

1. **使用 GPU 训练**：
   - 自动检测 CUDA 设备
   - 训练速度提升 10-50 倍

2. **调整学习率策略**：
   - ReduceLROnPlateau：验证损失停止下降时自动降低学习率
   - factor=0.5：每次降低为原来的一半
   - patience=5：连续5轮无改善触发调整

3. **数据增强策略**：
   - 提高模型泛化能力
   - 减少过拟合风险

## 🔍 常见问题

### Q: 为什么预测结果是黑白的，没有灰色过渡？

A: 检查是否正确保存概率图。当前版本输出的是原始概率图，包含灰色过渡区。

### Q: 如何提高分割精度？

A: 
1. 增加训练数据量（最多可使用30000张CelebAMask-HQ图片）
2. 延长训练时间（增加epochs）
3. 调整学习率（尝试 1e-4 或 1e-5）
4. 使用更多数据增强方法
5. 微调模型架构

### Q: 如何在边缘设备上部署？

A: 参考 `../unet_split/` 目录中的拆分模型实现：
1. 将模型拆分为编码器和解码器
2. 编码器部署在边缘设备（如Jetson）
3. 解码器部署在本地服务器
4. 通过Socket通信传输特征

## 📝 更新日志

### v3.0 - 2026-04-04
- ✅ 使用 CelebAMask-HQ 数据集（2000张图片）
- ✅ 实现数据增强（翻转、旋转、亮度/对比度、高斯噪声）
- ✅ 添加训练集/验证集划分（8:2比例）
- ✅ 实现学习率自动调整（ReduceLROnPlateau）
- ✅ 添加早停机制（10轮无改善自动停止）
- ✅ 保存最佳模型和检查点模型
- ✅ 支持切换不同轮次的模型进行预测

### v2.0 - 2026-03-20
- ✅ 优化模型架构，提升分割精度
- ✅ 修复负步长数组问题
- ✅ 优化训练流程和错误处理

### v1.0 - 2026-03-13
- ✅ 完成 PyTorch U-Net 架构实现
- ✅ 使用 Pascal VOC 数据集训练
- ✅ 实现概率图输出（保留灰色过渡区）
- ✅ 修复输出反转问题（人物=亮色）

## 🎯 下一步计划

- [ ] 添加 TensorRT 推理支持
- [ ] 实现实时摄像头分割
- [ ] 添加模型量化（FP16/INT8）
- [ ] 支持视频文件处理
- [ ] 添加 Web 界面演示
- [ ] 集成到边缘计算系统

## 📄 许可证

本项目基于原始 U-Net 实现修改，遵循相同的开源协议。

## 🙏 致谢

- 原始 U-Net 论文：[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- CelebAMask-HQ 数据集：[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
- PyTorch 框架：[PyTorch](https://pytorch.org/)
