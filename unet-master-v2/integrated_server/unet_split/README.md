# Jetson服务器 - WiFi设置与图像编码

## 项目说明
本项目实现了Jetson开发板上的WiFi参数设置和图像编码功能，通过统一的服务器提供API接口。

## 目录结构
```
unet_split/
├── jetson_combined_server.py  # 统一服务器主程序
├── wifi_controller.py         # WiFi管理控制器类
├── jetson_run_step.txt        # 运行步骤说明
└── saved_weights/             # 模型权重目录
    └── encoder_weights.pth    # U-Net编码器权重
```

## Jetson端依赖要求

### Python版本
- Python 3.10+

### 核心依赖包
```bash
# 基础依赖
pip install flask flask-cors torch torchvision numpy scikit-image opencv-python Pillow

# 系统工具（用于WiFi管理）
# 需要在Jetson主机上安装（非Docker环境）
sudo apt-get install wireless-tools network-manager
```

### 依赖版本说明
| 包名 | 版本 | 用途 |
|------|------|------|
| flask | 最新版本 | Web服务器框架 |
| flask-cors | 最新版本 | 跨域资源共享支持 |
| torch | 2.11.0+cu130 | PyTorch深度学习框架 |
| torchvision | 匹配torch版本 | 计算机视觉工具库 |
| numpy | 1.26.4 | 科学计算库 |
| scikit-image | 最新版本 | 图像处理库 |
| opencv-python | 最新版本 | OpenCV计算机视觉库 |
| Pillow | 最新版本 | 图像处理库 |

### 系统工具
| 工具 | 用途 |
|------|------|
| wireless-tools | 提供`iw`命令，用于WiFi参数查询和设置 |
| network-manager | 提供`nmcli`命令，用于网络连接管理 |

## 安装步骤

### 1. 上传文件到Jetson主机
将以下文件上传到`/home/nvidia/workspace`目录：
- jetson_combined_server.py
- wifi_controller.py
- saved_weights/encoder_weights.pth

### 2. 安装依赖
```bash
cd /home/nvidia/workspace

# 安装Python依赖
pip install flask flask-cors torch torchvision numpy scikit-image opencv-python Pillow

# 安装系统工具（需要sudo权限）
sudo apt-get update
sudo apt-get install wireless-tools network-manager

# 配置sudo免密码权限（用于WiFi设置）
echo "nvidia ALL=(ALL) NOPASSWD: /usr/bin/nmcli, /usr/bin/iw" | sudo tee /etc/sudoers.d/wifi-control
```

### 3. 运行服务器
```bash
python jetson_combined_server.py
```

## 服务说明

### WiFi设置功能（端口5000）
- **获取实际信道**: GET `/get_real_channel`
- **获取实际功率**: GET `/get_real_power`
- **获取连接设备数**: GET `/get_connected_devices`
- **设置信道**: POST `/set_channel` (参数: channel)
- **设置功率**: POST `/set_power` (参数: tx_power)

### 图像编码功能（Socket端口9000）
- 提供图像上传和编码处理服务
- 使用U-Net模型进行图像分割编码

## 前端连接
- WiFi设置: `http://<jetson_ip>:5000`
- 图像编码: Socket连接到端口9000

## 注意事项
1. 服务器需要在Jetson主机上直接运行（非Docker容器）
2. 需要配置sudo免密码权限才能正常设置WiFi参数
3. PyTorch在CPU模式下运行（如需GPU加速，需更新NVIDIA驱动）
4. 确保Jetson开发板已连接网络，IP地址可通过USB虚拟网口（192.168.55.1）或WiFi（10.42.0.1）访问
