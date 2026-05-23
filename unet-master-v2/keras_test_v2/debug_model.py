"""
调试模型预测
"""
import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from tensorflow.keras.models import load_model
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
print("     Model Prediction Debug")
print("=" * 70)
print()

# 加载模型
print("[1/4] Loading model...")
model_path = 'saved_models/keras_voc_unet.keras'
model = load_model(model_path)
print(f"Model loaded: {model_path}")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")
print()

# 加载一张测试图像
print("[2/4] Loading test image...")
test_path = '../person_segmentation/pascal_voc_person/images'
img_path = os.path.join(test_path, "00000.jpg")
img = io.imread(img_path)
print(f"Original image shape: {img.shape}")
print(f"Original image min: {img.min()}, max: {img.max()}")
print()

# 预处理
print("[3/4] Preprocessing image...")
if len(img.shape) == 2:
    img = np.stack([img, img, img], axis=2)
elif img.shape[2] == 4:
    img = img[:, :, :3]

img = img / 255
img = trans.resize(img, (256, 256))
img = np.reshape(img, (1,) + img.shape)
print(f"Processed image shape: {img.shape}")
print(f"Processed image min: {img.min()}, max: {img.max()}, mean: {img.mean()}")
print()

# 预测
print("[4/4] Making prediction...")
result = model.predict(img, verbose=1)
print(f"Prediction shape: {result.shape}")
print(f"Prediction min: {result.min():.6f}")
print(f"Prediction max: {result.max():.6f}")
print(f"Prediction mean: {result.mean():.6f}")
print(f"Prediction std: {result.std():.6f}")
print()

# 检查是否所有值相同
if np.allclose(result, result[0, 0, 0, 0]):
    print("WARNING: All predictions are the same value!")
    print(f"Constant value: {result[0, 0, 0, 0]:.6f}")
else:
    print("Prediction has variation.")
    
# 查看一些样本值
print("\nSample values from prediction:")
print(f"Top-left corner: {result[0, 0, 0, 0]:.6f}")
print(f"Center: {result[0, 128, 128, 0]:.6f}")
print(f"Bottom-right corner: {result[0, 255, 255, 0]:.6f}")
