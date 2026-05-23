"""
Keras U-Net Pascal VOC 人物分割训练脚本
使用原始Keras架构训练模型
"""
import sys
import os
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
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
def dice_loss(y_true, y_pred):
    smooth = 1e-5
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice

print("=" * 70)
print("     Keras U-Net Pascal VOC Person Segmentation Training")
print("=" * 70)
print()

# 定义U-Net模型
def unet_rgb(input_size=(256, 256, 3)):
    """原始Keras U-Net模型定义"""
    inputs = Input(input_size)
    
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
    d4 = Dropout(0.5)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(d4)
    
    # 瓶颈层
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    d5 = Dropout(0.5)(c5)
    
    # 解码器
    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(d5)
    u6 = concatenate([u6, d4])
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

# 数据处理函数
def adjustData(img, mask):
    """调整图像和掩码数据"""
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, 
                   image_color_mode="rgb", mask_color_mode="grayscale", 
                   image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, 
                   target_size=(256, 256), seed=1):
    """数据生成器"""
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed
    )
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed
    )
    
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)
        yield (img, mask)

# 数据增强参数
data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

print("[1/3] Setting up data generator...")
train_path = '../person_segmentation/pascal_voc_person'

# 检查数据是否存在
if not os.path.exists(train_path):
    print(f"✗ Dataset not found: {train_path}")
    sys.exit(1)

images_dir = os.path.join(train_path, 'images')
masks_dir = os.path.join(train_path, 'masks')

if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
    print("✗ Images or masks directory not found!")
    sys.exit(1)

# 计算训练图片数量
train_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
num_images = len(train_images)

print(f"  Found {num_images} training images")

if num_images == 0:
    print("✗ No training data found!")
    sys.exit(1)

# 限制训练图片数量为前100张
max_train_images = 100
if num_images > max_train_images:
    print(f"  Using first {max_train_images} images")
    num_images = max_train_images

myGene = trainGenerator(
    batch_size=2,
    train_path=train_path,
    image_folder='images',
    mask_folder='masks',
    aug_dict=data_gen_args,
    save_to_dir=None
)

print("✓ Data generator ready")
print()

print("[2/3] Creating and training model...")
model = unet_rgb()

# 模型检查点
model_checkpoint = ModelCheckpoint(
    'saved_models/keras_voc_unet.keras',
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
num_epochs = 10

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
print("Model saved as: saved_models/keras_voc_unet.keras")
print()
print("Next step: Run predict_keras_voc.py to test the model")
print()
