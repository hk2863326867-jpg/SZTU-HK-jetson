"""
使用CelebAMask-HQ训练的U-Net模型预测VOC数据集效果
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import skimage.io as io
import skimage.transform as trans
from tensorflow.keras.models import load_model

print("=" * 70)
print("     Predict VOC Dataset with CelebAMask-HQ Model")
print("=" * 70)
print()

def testGenerator(test_path, start_idx=0, end_idx=10, target_size=(256, 256)):
    """测试数据生成器"""
    # VOC数据集图片命名格式：00000.jpg, 00001.jpg, ...
    for i in range(start_idx, end_idx):
        img_path = os.path.join(test_path, f"{i:05d}.jpg")
        if not os.path.exists(img_path):
            print(f"Image {i} not found, skipping...")
            continue
        
        img = io.imread(img_path)
        
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, (1,) + img.shape)
        yield (img,)

def saveResult(save_path, npyfile, start_idx=0):
    """保存预测结果"""
    os.makedirs(os.path.join(save_path, "raw"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "thresholded"), exist_ok=True)
    
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        img_idx = start_idx + i
        
        # 保存原始概率图
        raw_img = (img * 255).astype(np.uint8)
        io.imsave(os.path.join(save_path, "raw", "%05d_raw.png" % img_idx), raw_img)
        
        # 应用阈值处理 (使用0.7阈值，与CelebAMask-HQ预测相同)
        thresholded = (img > 0.7).astype(np.float32)
        thresholded_img = (thresholded * 255).astype(np.uint8)
        io.imsave(os.path.join(save_path, "thresholded", "%05d_thresholded.png" % img_idx), thresholded_img)
        
        # 打印统计信息
        print(f"Image {img_idx:05d}: min={img.min():.4f}, max={img.max():.4f}, mean={img.mean():.4f}")

print("[1/3] Loading model...")
model_path = 'saved_models/celebamask_unet.keras'

if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    print("Please run train_celebamask.py first.")
    sys.exit(1)

model = load_model(model_path)
print("  Model loaded successfully!")

print("\n[2/3] Setting up test data...")
test_path = '../person_segmentation/pascal_voc_person/images'
output_dir = '../pytorch_unet/results_voc_with_celeb_model'

if not os.path.exists(test_path):
    print(f"Test data not found: {test_path}")
    print("Please make sure VOC dataset is available.")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

# 使用编号00000到00010的图片（VOC数据集）
test_images = testGenerator(test_path, start_idx=0, end_idx=11)

print("\n[3/3] Making predictions...")
results = model.predict(test_images, verbose=1)

print("\nSaving results...")
saveResult(output_dir, results, start_idx=0)

print()
print("=" * 70)
print("Prediction complete!")
print("=" * 70)
print(f"Results saved to: {output_dir}/")
