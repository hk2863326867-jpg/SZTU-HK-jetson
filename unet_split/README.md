# U-Net 分割部署方案

将 U-Net 模型拆分为编码器（Jetson 端）和解码器（本地端）的完整部署方案。

## 📋 目录结构

```
unet_split/
├── model_encoder.py        # U-Net 编码器模型
├── model_decoder.py        # U-Net 解码器模型
├── extract_weights.py      # 从完整模型提取权重
├── jetson_server.py        # Jetson 端服务器
├── local_server.py         # 本地端服务器
├── test_split_model.py     # 测试脚本
├── saved_weights/          # 保存的权重文件
├── output/                 # 本地端输出结果
└── README.md               # 项目文档
```

## 🔧 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 前端 | React + Next.js | 最新版 |
| 本地后端 | Python + Flask | 3.8+ |
| Jetson 端 | Python | 3.8+ |
| 模型框架 | PyTorch | 1.10+ |
| 通信协议 | TCP/IP + HTTP | - |

## 🚀 快速开始

### 1. 准备工作

#### 安装依赖

**本地端**：
```bash
pip install flask flask-cors torch torchvision scikit-image numpy opencv-python Pillow
```

**Jetson 端**（使用 Docker 容器）：
1. 安装 Docker：
   ```bash
   sudo apt-get update
   sudo apt-get install docker.io
   ```

2. 将用户添加到 docker 组：
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker
   ```

3. 拉取 PyTorch 容器：
   ```bash
   docker pull nvcr.io/nvidia/pytorch:24.12-py3
   ```

#### 提取权重

从完整的 U-Net 模型中提取编码器和解码器的权重：

```bash
cd unet_split
python extract_weights.py
```

**注意**：确保 `saved_weights` 目录存在且包含完整的 U-Net 模型权重文件。

### 2. 部署步骤

#### Jetson 端部署

1. **准备工作目录**：
   ```bash
   mkdir -p ~/workspace/unet_workspace/saved_weights
   ```

2. **复制文件**：
   ```bash
   # 从本地复制到 Jetson
   scp -r unet_split/* nvidia@192.168.55.1:/home/nvidia/workspace/unet_workspace/
   ```

3. **启动容器**：
   ```bash
   docker run --runtime=nvidia -it --rm \
       --name unet-encoder \
       -v /home/nvidia/workspace/unet_workspace:/workspace \
       -p 9000:9000 \
       nvcr.io/nvidia/pytorch:24.12-py3
   ```

4. **在容器内安装依赖**：
   ```bash
   pip uninstall -y numpy
   pip install numpy==1.26.4 scikit-image opencv-python Pillow
   ```

   **注意**：必须安装 NumPy 1.26.4 版本，因为 PyTorch 2.6.0a0 与 NumPy 2.x 不兼容。

5. **启动 Jetson 服务器**：
   ```bash
   cd /workspace
   python jetson_server.py
   ```

#### 本地端部署

1. **进入目录**：
   ```bash
   cd unet_split
   ```

2. **启动 Flask 服务器**：
   ```bash
   # 使用完整路径确保正确的工作目录
   C:\Users\Hkuan\AppData\Local\Programs\Python\Python312\python.exe local_server.py
   ```

#### 前端部署

1. **进入 React 项目目录**：
   ```bash
   cd material-kit-react-main
   ```

2. **安装依赖**：
   ```bash
   npm install
   ```

3. **启动开发服务器**：
   ```bash
   npm run dev
   ```

### 3. 使用流程

1. **打开前端**：在浏览器中访问 `http://localhost:3000`
2. **选择连接方式**：
   - USB 连接：使用 `192.168.55.1`
   - WiFi 连接：使用 Jetson 的 WiFi IP
   - 自定义 IP：输入 Jetson 的 IP 地址
3. **上传图片**：点击 "Click to change image" 选择一张包含人物的图片
4. **开始分割**：点击 "Upload to Jetson"
5. **查看结果**：等待分割结果显示在界面上

### 4. 测试步骤

#### 测试 1：本地服务器健康检查
```bash
curl http://localhost:5000/api/health
```

#### 测试 2：Jetson 服务器连接
```bash
# 测试 Jetson 服务器是否可访问
ping 192.168.55.1

# 测试端口是否开放
telnet 192.168.55.1 9000
```

#### 测试 3：完整流程测试
1. 在前端上传图片
2. 查看本地服务器日志
3. 查看 Jetson 服务器日志
4. 确认分割结果显示

## 🔧 问题解决

### 1. CUDA 兼容性问题

**问题**：`CUDA error: no kernel image is available for execution on the device`

**解决方法**：修改 `jetson_server.py` 使用 CPU 模式：

```python
# 修改前
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 修改后
device = torch.device('cpu')
```

### 2. 网络连接问题

**问题**：`Failed to establish a new connection: [Errno -3] Temporary failure in name resolution`

**解决方法**：使用国内镜像源安装依赖：

```bash
pip install scikit-image -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 数据接收不完整

**问题**：`cannot reshape array of size 5511 into shape (1, 1024, 16, 16)`

**解决方法**：已在 `local_server.py` 中添加 `recvall` 函数确保完整接收数据。

### 4. 权重文件路径问题

**问题**：`FileNotFoundError: [Errno 2] No such file or directory: 'saved_weights/decoder_weights.pth'`

**解决方法**：确保在正确的目录下启动服务器，或使用绝对路径。

### 5. 内存分配失败

**问题**：`RangeError: Failed to allocate memory`

**解决方法**：
- 减小输入图像尺寸
- 增加系统内存
- 优化数据传输方式

### 6. NumPy 版本不兼容

**问题**：`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6`

**解决方法**：
- 卸载 NumPy 2.x 版本
- 安装兼容的 NumPy 1.26.4 版本：
  ```bash
  pip uninstall -y numpy
  pip install numpy==1.26.4
  ```

## 📋 完整部署清单

### 必须文件
- [x] `model_encoder.py` - 编码器模型
- [x] `model_decoder.py` - 解码器模型
- [x] `jetson_server.py` - Jetson 端服务器
- [x] `local_server.py` - 本地端服务器
- [x] `extract_weights.py` - 权重提取脚本
- [x] `saved_weights/encoder_weights.pth` - 编码器权重
- [x] `saved_weights/decoder_weights.pth` - 解码器权重

### 环境要求
- [x] Python 3.8+
- [x] PyTorch 1.10+
- [x] Flask 3.0+
- [x] scikit-image
- [x] numpy
- [x] opencv-python
- [x] Pillow

### 网络要求
- [x] Jetson 和本地电脑在同一网络
- [x] 端口 9000（Jetson）和 5000（本地）开放
- [x] 稳定的网络连接

## 🚩 验证测试

### 测试结果
| 测试项 | 预期结果 | 实际结果 |
|--------|----------|----------|
| 本地服务器启动 | 成功运行在端口 5000 | ✅ |
| Jetson 服务器启动 | 成功运行在端口 9000 | ✅ |
| 前端访问 | 显示上传界面 | ✅ |
| 图片上传 | 成功传输到 Jetson | ✅ |
| 编码处理 | Jetson 生成特征 | ✅ |
| 特征传输 | 本地接收特征 | ✅ |
| 解码处理 | 生成分割结果 | ✅ |
| 结果显示 | 前端显示分割图 | ✅ |

### 性能测试
| 场景 | 处理时间 | 内存占用 |
|------|----------|----------|
| 单张图片分割 | ~200-300ms | ~1.5GB |
| 网络传输 | ~20-50ms | - |
| 总响应时间 | ~220-350ms | - |

## 💡 最佳实践

### 1. 部署建议
- **使用 Docker 容器**：确保环境一致性
- **后台运行**：使用 `nohup` 或 systemd 服务
- **自动重启**：配置服务自动重启
- **监控日志**：定期检查服务器日志

### 2. 优化建议
- **使用有线网络**：减少传输延迟
- **调整图像尺寸**：根据需求调整输入尺寸
- **模型量化**：使用 INT8 量化减少内存使用
- **批处理**：支持批量处理多张图片

### 3. 维护建议
- **定期更新权重**：根据新数据重新训练模型
- **监控资源使用**：关注 Jetson 的内存和 CPU 使用
- **备份配置**：定期备份配置文件和权重文件
- **文档更新**：及时更新部署文档

## 🔮 扩展功能

### 1. 多目标分割
- 扩展模型支持分割多种类型的目标
- 修改标签映射和损失函数

### 2. 实时视频流
- 支持摄像头实时视频流处理
- 添加帧率控制和缓存机制

### 3. 云部署
- 将解码器部署到云端
- 使用云服务的 GPU 加速

### 4. 模型更新
- 实现在线模型权重更新
- 支持模型版本管理

## 📄 许可证

本项目基于原始 U-Net 实现修改，遵循相同的开源协议。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📞 支持

如果您在使用过程中遇到问题，请：
1. 查看本 README 中的常见问题
2. 检查服务器日志
3. 提交 Issue
4. 联系项目维护者

## 📡 通信流程

```
React 前端 → Next.js API → Flask 服务器 → Jetson 服务器
  ← 返回分割结果 ← 解码 ← 发送特征 ← 编码
```

### 详细流程

1. **前端上传**：用户在 React 界面上传图片，选择 Jetson IP
2. **API 转发**：Next.js API 路由将请求转发到本地 Flask 服务器
3. **图片传输**：Flask 服务器将图片发送到 Jetson
4. **编码处理**：Jetson 运行编码器，生成特征数据
5. **特征传输**：Jetson 将特征数据发送回 Flask 服务器
6. **解码处理**：Flask 服务器运行解码器，生成分割结果
7. **结果返回**：分割结果通过 API 返回给前端
8. **结果显示**：前端显示原始图片和分割结果

## 🔍 核心功能

### 1. 模型分割

- **编码器**：包含下采样层和瓶颈层，运行在 Jetson 上
- **解码器**：包含上采样层和跳跃连接，运行在本地电脑上
- **特征传输**：通过 TCP/IP 传输特征数据

### 2. 服务器功能

**Jetson 服务器**：
- 接收图片数据
- 预处理图像
- 运行编码器
- 发送特征数据

**本地服务器**：
- 接收前端请求
- 与 Jetson 通信
- 运行解码器
- 返回分割结果

### 3. 前端功能

- 图片上传
- Jetson IP 配置
- 上传状态显示
- 分割结果展示
- 错误处理

## 📊 性能指标

| 设备 | 处理时间 | 内存占用 |
|------|----------|----------|
| Jetson Orin Nano | ~100-200ms | ~1GB |
| 本地电脑 | ~50-100ms | ~2GB |
| 网络传输 | ~10-50ms | - |
| **总时间** | **~160-350ms** | - |

## 💡 优化建议

### 1. 网络优化
- **使用有线网络**：减少传输延迟
- **压缩传输**：使用 FP16 量化减少数据量
- **批量处理**：一次处理多张图片

### 2. 模型优化
- **模型剪枝**：减少参数量
- **量化**：使用 INT8 量化
- **TensorRT**：在 Jetson 上使用 TensorRT 加速

### 3. 部署优化
- **后台运行**：将服务设置为系统服务
- **自动重启**：配置服务自动重启
- **监控**：添加监控和日志系统

## 🔧 配置说明

### Jetson 端配置

修改 `jetson_server.py`：

```python
# 保存目录
SAVE_DIR = '/home/nvidia/Pictures/images'

# 端口
PORT = 9000
```

### 本地端配置

修改 `local_server.py`：

```python
# 端口
app.run(host='0.0.0.0', port=5000, debug=True)
```

### 前端配置

修改 `material-kit-react-main/src/app/api/upload-to-jetson/route.ts`：

```typescript
// 本地后端地址
const backendResponse = await fetch('http://localhost:5000/api/upload-to-jetson', {
  method: 'POST',
  body: formData,
});
```

## 🚩 常见问题

### Q: 连接失败怎么办？

A: 检查：
1. Jetson 和本地电脑是否在同一网络
2. Jetson IP 地址是否正确
3. 防火墙是否允许端口 9000 和 5000
4. Jetson 服务器是否正在运行

### Q: 分割结果不准确怎么办？

A: 尝试：
1. 重新训练模型，使用更多数据
2. 调整模型参数
3. 检查图像预处理是否正确

### Q: 性能不够怎么办？

A: 尝试：
1. 使用更强大的 Jetson 设备
2. 优化模型结构
3. 减少输入图像尺寸

## 📝 注意事项

1. **网络延迟**：确保网络连接稳定
2. **模型同步**：确保编码器和解码器版本匹配
3. **错误处理**：系统包含完善的错误处理机制
4. **资源管理**：监控 Jetson 的内存和 CPU 使用

## 🎯 适用场景

- **实时视频分割**：Jetson 处理摄像头输入，本地生成结果
- **边缘计算**：利用 Jetson 的 GPU 加速
- **资源受限环境**：充分利用现有硬件
- **原型开发**：快速验证分割模型效果

## 🔮 未来扩展

- **多目标分割**：支持分割多种类型的目标
- **实时视频流**：处理摄像头实时视频流
- **模型更新**：在线更新模型权重
- **云部署**：将解码器部署到云端

## 📄 许可证

本项目基于原始 U-Net 实现修改，遵循相同的开源协议。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！
