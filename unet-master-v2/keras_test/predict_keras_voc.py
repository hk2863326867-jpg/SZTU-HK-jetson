"""
Keras U-Net Pascal VOC 人物分割预测脚本
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
print("     Keras U-Net Pascal VOC Prediction")
print("=" * 70)
print()

def testGenerator(test_path, start_idx=100, end_idx=110, target_size=(256, 256), as_gray=False):
    """测试数据生成器 - 只预测第100-110张图片"""
    for i in range(start_idx, end_idx):
        img_path = os.path.join(test_path, "%05d.jpg" % i)
        if not os.path.exists(img_path):
            continue
        
        if as_gray:
            img = io.imread(img_path, as_gray=True)
        else:
            img = io.imread(img_path)
        
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, (1,) + img.shape)
        yield (img,)

def saveResult(save_path, npyfile, start_idx=100, flag_multi_class=False, num_class=2):
    """保存预测结果"""
    os.makedirs(os.path.join(save_path, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "thresholded"), exist_ok=True)
    
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        img_idx = start_idx + i
        
        # 保存原始概率图
        raw_img = (img * 255).astype(np.uint8)
        io.imsave(os.path.join(save_path, "raw", "%05d_raw.png" % img_idx), raw_img)
        
        # 应用阈值处理 (降低阈值到0.3)
        thresholded = (img > 0.3).astype(np.float32)
        thresholded_img = (thresholded * 255).astype(np.uint8)
        io.imsave(os.path.join(save_path, "thresholded", "%05d_thresholded.png" % img_idx), thresholded_img)
        
        # 打印统计信息
        print(f"Image {img_idx:05d}: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")

print("[1/3] Loading model...")
model_path = 'saved_models/keras_voc_unet.keras'

if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    print("Please run train_keras_voc.py first.")
    sys.exit(1)

model = load_model(model_path)
print("  Model loaded successfully!")

print("\n[2/3] Setting up test data...")
test_path = '../person_segmentation/pascal_voc_person/images'
output_dir = 'results_keras_voc'

if not os.path.exists(test_path):
    print(f"Test data not found: {test_path}")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

# 使用编号100-110的图片（共10张）
test_images = testGenerator(test_path, start_idx=100, end_idx=110)

print("\n[3/3] Making predictions...")
results = model.predict(test_images, verbose=1)

print("\nSaving results...")
saveResult(output_dir, results, start_idx=100)

print()
print("=" * 70)
print("Prediction complete!")
print("=" * 70)
print(f"Results saved to: {output_dir}/")
