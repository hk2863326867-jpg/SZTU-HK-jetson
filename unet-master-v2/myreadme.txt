================================================================================
                        U-Net 项目修改与使用说明
================================================================================

项目名称：U-Net 医学图像分割（ISBI 神经元膜分割数据集）
修改日期：2026-02-17
修改者：本地适配

================================================================================
一、项目概述
================================================================================

原始项目：GitHub 上的 Keras U-Net 实现（2018年左右）
项目目的：使用 U-Net 网络对生物医学图像进行语义分割
数据集：ISBI 神经元膜分割数据集（30张训练图，30张测试图）

================================================================================
二、代码修改日志
================================================================================

【文件 1：data.py】
修改位置：第 2 行
原代码：from keras.preprocessing.image import ImageDataGenerator
新代码：from tensorflow.keras.preprocessing.image import ImageDataGenerator
修改原因：Keras 3.x 中，keras.preprocessing 被移到 tensorflow.keras.preprocessing

【文件 2：model.py】
修改位置：第 6-10 行
原代码：
  from keras.models import *
  from keras.layers import *
  from keras.optimizers import *
  from keras.callbacks import ModelCheckpoint, LearningRateScheduler
  from keras import backend as keras

新代码：
  from tensorflow.keras.models import *
  from tensorflow.keras.layers import *
  from tensorflow.keras.optimizers import *
  from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
  from tensorflow.keras import backend as keras

修改原因：统一使用 tensorflow.keras 导入，兼容 Keras 3.x

修改位置：第 55 行
原代码：model = Model(input = inputs, output = conv10)
新代码：model = Model(inputs = inputs, outputs = conv10)
修改原因：Model 构造器参数名从 input/output 改为 inputs/outputs

修改位置：第 57 行
原代码：model.compile(optimizer = Adam(lr = 1e-4), ...)
新代码：model.compile(optimizer = Adam(learning_rate = 1e-4), ...)
修改原因：Adam 优化器参数名从 lr 改为 learning_rate

【文件 3：main.py】
修改位置：第 17 行
原代码：model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', ...)
新代码：model_checkpoint = ModelCheckpoint('unet_membrane.keras', ...)
修改原因：Keras 3.x 不再支持 .hdf5 格式，必须使用 .keras 格式

修改位置：第 18 行
原代码：model.fit_generator(...)
新代码：model.fit(...)
修改原因：fit_generator 已被弃用，直接使用 fit

修改位置：第 21 行
原代码：results = model.predict_generator(testGene, 30, verbose=1)
新代码：results = model.predict(testGene, 30, verbose=1)
修改原因：predict_generator 已被弃用，直接使用 predict

================================================================================
三、新增文件说明
================================================================================

【full_run.py】
功能：完整的训练+预测+保存流程
使用场景：第一次运行，或需要重新训练模型
运行命令：python full_run.py
运行时间：10-30 分钟（取决于电脑性能）

【quick_predict.py】
功能：使用已训练模型进行预测和保存
使用场景：已有 unet_membrane.keras 模型，只想预测新图像
运行命令：python quick_predict.py
运行时间：1-2 分钟
前置条件：必须先运行 full_run.py 生成模型文件

【predict.py】（已废弃）
功能：早期尝试的预测脚本
状态：保存图像时出错，已被 save_results.py 替代

【save_results.py】（已废弃）
功能：修复后的预测脚本
状态：功能正常，但已被 quick_predict.py 替代

【run_as_admin.bat】（辅助文件）
功能：Windows 批处理脚本，一键安装依赖并运行
使用方式：右键 -> 以管理员身份运行

【run.ps1】（辅助文件）
功能：PowerShell 脚本，安装依赖并运行
使用方式：在管理员 PowerShell 中运行

================================================================================
四、运行步骤
================================================================================

【步骤 1：环境准备】
1. 确保已安装 Python 3.10 或更高版本
2. 打开管理员 PowerShell（Win + X -> Windows PowerShell (管理员)）
3. 进入项目目录：
   cd d:\OneDrive\桌面\github\unet-master

【步骤 2：安装依赖】
运行以下命令安装所需库：
   pip install tensorflow scikit-image

或者使用批处理脚本：
   右键点击 run_as_admin.bat -> 以管理员身份运行

【步骤 3：训练模型（首次运行）】
运行完整训练流程：
   python full_run.py

预期输出：
  - 数据加载信息
  - 训练进度（300 steps）
  - 模型保存提示
  - 预测进度
  - 结果保存提示

运行时间：10-30 分钟

【步骤 4：预测新图像（可选）】
如果已有训练好的模型，只需预测：
   python quick_predict.py

运行时间：1-2 分钟

【步骤 5：查看结果】
打开文件夹：data\membrane\test\
对比查看：
  - 0.png - 29.png：原始测试图像
  - 0_predict.png - 29_predict.png：U-Net 预测结果

================================================================================
五、训练参数说明
================================================================================

【网络结构】
- 输入尺寸：256x256x1（灰度图）
- 输出尺寸：256x256x1（二分类概率图）
- 激活函数：最后一层使用 sigmoid（输出 0-1 之间的概率）

【训练参数】
- Batch size：2（每次处理 2 张图像）
- Steps per epoch：300（每个 epoch 处理 600 张图，含数据增强）
- Epochs：1（完整遍历数据集 1 次）
- 可调整：修改 full_run.py 中的 epochs 参数，如 epochs=10

【损失函数】
- Binary Crossentropy（二分类交叉熵）
- 适用于二分类问题（膜 vs 背景）

【优化器】
- Adam 优化器
- 学习率：0.0001（1e-4）

【评估指标】
- Accuracy（准确率）

【数据增强】
- 旋转范围：±20%
- 宽度/高度平移：±5%
- 剪切范围：±5%
- 缩放范围：±5%
- 水平翻转：是
- 填充模式：nearest

================================================================================
六、注意事项
================================================================================

【1. 版本兼容性】
- 本项目已适配 Keras 3.x + TensorFlow 2.x
- 原始代码使用 Keras 2.x，API 已发生变化
- 如需使用原始代码，需安装旧版本：
  pip install tensorflow==2.3 keras==2.3

【2. 模型文件】
- 训练完成后会生成：unet_membrane.keras
- 不要删除此文件，后续预测需要用到
- 如需重新训练，删除此文件即可

【3. 测试图像要求】
- 必须是灰度图像（单通道）
- 建议尺寸：512x512 左右（会自动调整到 256x256）
- 命名格式：0.png, 1.png, ..., 29.png
- 图像内容：与训练数据相似（电子显微镜图像效果最佳）

【4. 训练时间】
- 1 epoch：10-30 分钟
- 10 epochs：1-2 小时
- 更多 epochs 可以提高模型性能，但需要更多时间

【5. 预测结果差异】
- 每次训练结果会有细微差异（正常现象）
- 原因：随机初始化、数据增强随机性、Dropout 等
- 如需稳定结果，设置随机种子：
  import numpy as np
  np.random.seed(42)
  import tensorflow as tf
  tf.random.set_seed(42)

【6. Windows 权限问题】
- 如果遇到权限错误，必须使用管理员 PowerShell
- 或使用 run_as_admin.bat 以管理员身份运行

【7. GPU 加速（可选）】
- 默认使用 CPU 训练（速度较慢）
- 如有 NVIDIA GPU，可安装 GPU 版本 TensorFlow：
  pip uninstall tensorflow
  pip install tensorflow-gpu
- 需要安装 CUDA 和 cuDNN

【8. 内存要求】
- 训练时内存占用约 2-4 GB
- 确保至少有 8 GB 可用内存

================================================================================
七、常见问题解决
================================================================================

【问题 1：ModuleNotFoundError: No module named 'keras'】
解决：安装 Keras
  pip install keras

【问题 2：ValueError: The filepath provided must end in `.keras`】
解决：已修复，确保使用 .keras 扩展名

【问题 3：OSError: cannot write mode F as PNG】
解决：已修复，图像保存前转换为 uint8 类型

【问题 4：训练速度很慢】
解决：
  - 使用 GPU 版本 TensorFlow
  - 减少 steps_per_epoch
  - 减小 batch_size

【问题 5：预测结果全黑或全白】
解决：
  - 检查测试图像是否为灰度图
  - 确保模型已正确加载
  - 尝试重新训练模型

【问题 6：权限错误（Permission denied）】
解决：使用管理员 PowerShell 运行

================================================================================
八、项目成果
================================================================================

【训练结果】
- 训练集准确率：83.34%
- 最终损失：0.3619
- 训练时间：约 30 分钟（1 epoch）

【预测结果】
- 成功生成 30 张预测图像
- 保存在：data\membrane\test\*_predict.png
- 模型成功识别出神经元膜结构

【技术收获】
- 理解 U-Net 网络架构（编码器-解码器-跳跃连接）
- 掌握 Keras/TensorFlow 的使用
- 学习数据增强技术
- 掌握深度学习训练流程

================================================================================
九、后续改进方向
================================================================================

【1. 增加训练轮数】
- 从 1 epoch 增加到 10-50 epochs
- 预期：准确率提升到 90%+

【2. 调整超参数】
- 学习率：尝试 1e-3, 1e-5
- Batch size：尝试 4, 8
- 数据增强参数：调整旋转、平移范围

【3. 改进网络结构】
- 添加更多卷积层
- 使用残差连接
- 尝试注意力机制

【4. 扩展数据集】
- 收集更多训练图像
- 使用其他医学图像数据集

【5. 尝试其他任务】
- 准备自己的数据集（图像 + 标注）
- 重新训练模型
- 应用于其他分割任务

================================================================================
十、文件清单
================================================================================

【核心文件】
- model.py          - U-Net 网络定义
- data.py           - 数据处理函数
- main.py           - 原始主程序（已修复）

【运行脚本】
- full_run.py       - 完整训练+预测流程（推荐）
- quick_predict.py   - 快速预测脚本（推荐）

【辅助文件】
- run_as_admin.bat  - Windows 批处理脚本
- run.ps1          - PowerShell 脚本
- myreadme.txt      - 本文件

【数据目录】
- data/membrane/train/image/     - 训练图像（30张）
- data/membrane/train/label/     - 训练标注（30张）
- data/membrane/test/           - 测试图像（30张）
- data/membrane/test/*_predict.png - 预测结果（30张）

【模型文件】
- unet_membrane.keras  - 训练好的模型

================================================================================
十一、联系方式与参考
================================================================================

【原始项目】
- GitHub: https://github.com/zhixuhao/unet
- 作者: zhixuhao

【U-Net 论文】
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional
  Networks for Biomedical Image Segmentation. MICCAI 2015.

【ISBI 数据集】
- ISBI 2012 EM Segmentation Challenge
- http://brainiac2.mit.edu/isbi_challenge/home/

【Keras 文档】
- https://keras.io/

【TensorFlow 文档】
- https://www.tensorflow.org/

================================================================================
                            文档结束
================================================================================
