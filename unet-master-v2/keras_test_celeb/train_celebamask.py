"""
Keras U-Net CelebAMask-HQ 人脸分割训练脚本
使用CelebAMask-HQ数据集进行人脸分割训练
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob

print("=" * 70)
print("     Keras U-Net CelebAMask-HQ Face Segmentation Training")
print("=" * 70)
print()

# 数据增强配置
data_gen_args = dict(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def unet_rgb():
    """RGB输入的U-Net模型定义"""
    inputs = Input((256, 256, 3))
    
    # 编码器
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    
    # 瓶颈层
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # 解码器
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    # 输出层
    out1 = Conv2D(2, (3, 3), activation='relu', padding='same')(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(out1)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def load_celebamask_data(img_idx, img_dir, mask_dir):
    """加载CelebAMask-HQ数据"""
    # 加载图片
    img_path = os.path.join(img_dir, f"{img_idx}.jpg")
    img = io.imread(img_path)
    
    # 加载所有部位的掩码并合并
    mask_parts = []
    subfolder = str(img_idx // 2000)  # 每2000张图片一个文件夹
    mask_subdir = os.path.join(mask_dir, subfolder)
    
    # 查找所有掩码文件
    mask_files = glob.glob(os.path.join(mask_subdir, f"{img_idx:05d}_*.png"))
    
    if mask_files:
        print(f"  Found {len(mask_files)} mask parts for image {img_idx}")
        for mask_file in mask_files:
            # 读取为灰度图
            mask = io.imread(mask_file, as_gray=True)
            # 确保掩码是0-1范围
            if mask.max() > 1.0:
                mask = mask / 255.0
            mask_parts.append(mask)
        
        # 合并所有部位为一个掩码
        mask = np.max(np.stack(mask_parts), axis=0)
    else:
        print(f"  No masks found for image {img_idx}, using empty mask")
        # 如果没有掩码，创建全黑掩码
        mask = np.zeros_like(img[:, :, 0])
    
    return img, mask

def trainGenerator(batch_size, img_dir, mask_dir, aug_dict, num_images=100):
    """训练数据生成器"""
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    batch_images = []
    batch_masks = []
    
    while True:
        for img_idx in range(num_images):
            try:
                img, mask = load_celebamask_data(img_idx, img_dir, mask_dir)
                
                # 预处理
                img = img / 255.0
                # 掩码已经是0-1范围，不需要再除以255
                
                # 调整大小（图片是1024x1024，掩码是512x512）
                img = trans.resize(img, (256, 256))
                mask = trans.resize(mask, (256, 256))
                
                # 扩展维度
                mask = np.expand_dims(mask, axis=-1)
                
                # 数据增强
                img_aug = image_datagen.random_transform(img)
                mask_aug = mask_datagen.random_transform(mask)
                
                batch_images.append(img_aug)
                batch_masks.append(mask_aug)
                
                # 当批次满了就yield
                if len(batch_images) == batch_size:
                    yield (np.array(batch_images), np.array(batch_masks))
                    batch_images = []
                    batch_masks = []
                    
            except Exception as e:
                print(f"Error loading image {img_idx}: {e}")
                continue
        
        # 如果最后一批不满batch_size，也yield
        if batch_images:
            yield (np.array(batch_images), np.array(batch_masks))
            batch_images = []
            batch_masks = []

# 设置数据集路径
celeb_root = '../CelebAMask-HQ'
img_dir = os.path.join(celeb_root, 'CelebA-HQ-img')
mask_dir = os.path.join(celeb_root, 'CelebAMask-HQ-mask-anno')

print("[1/3] Setting up data generator...")

# 检查数据是否存在
if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
    print(f"✗ Dataset not found!")
    print(f"  Image dir: {img_dir}")
    print(f"  Mask dir: {mask_dir}")
    sys.exit(1)

# 计算训练图片数量
img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
num_images = len(img_files)

print(f"  Found {num_images} training images")

if num_images == 0:
    print("✗ No training data found!")
    sys.exit(1)

# 限制训练图片数量为前100张（快速验证效果）
max_train_images = 100
if num_images > max_train_images:
    print(f"  Using first {max_train_images} images")
    num_images = max_train_images

myGene = trainGenerator(
    batch_size=2,
    img_dir=img_dir,
    mask_dir=mask_dir,
    aug_dict=data_gen_args,
    num_images=num_images
)

print("✓ Data generator ready")
print()

print("[2/3] Creating and training model...")
model = unet_rgb()

# 模型检查点
model_checkpoint = ModelCheckpoint(
    'saved_models/celebamask_unet.keras',
    monitor='loss',
    verbose=1,
    save_best_only=True
)

# 早停
early_stopping = EarlyStopping(
    monitor='loss',
    patience=5,
    verbose=1
)

# 计算训练参数
steps_per_epoch = min(num_images // 2, 50)
num_epochs = 5

print("Starting training...")
print(f"  Batch size: 2")
print(f"  Steps per epoch: {steps_per_epoch}")
print(f"  Epochs: {num_epochs}")
print(f"  Training images: {num_images}")
print(f"  Input size: 256x256x3 (RGB)")
print()

# 创建保存目录
os.makedirs('saved_models', exist_ok=True)

# 训练模型
model.fit(
    myGene,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    callbacks=[model_checkpoint, early_stopping]
)

print()
print("=" * 70)
print("✓ Training complete!")
print("=" * 70)
print()
print("Model saved as: saved_models/celebamask_unet.keras")
print()
print("Next step: Run predict_celebamask.py to test the model")
print()
