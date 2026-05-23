"""
下载 COCO 数据集中的人物分割数据
只下载人物类别的图片和标注
"""
import os
import sys
import urllib.request
import json
import zipfile

print("=" * 70)
print("     Downloading COCO Person Dataset")
print("=" * 70)
print()

# 创建目录
dataset_dir = "coco_person"
images_dir = os.path.join(dataset_dir, "images")
masks_dir = os.path.join(dataset_dir, "masks")

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

print("This script will download person images from COCO dataset.")
print()
print("Options:")
print("  1. Download 100 images (fast, ~500MB)")
print("  2. Download 500 images (medium, ~2GB)")
print("  3. Download 1000 images (slow, ~4GB)")
print()

choice = input("Enter your choice (1/2/3) [default=1]: ").strip()
if choice == '2':
    num_images = 500
elif choice == '3':
    num_images = 1000
else:
    num_images = 100

print()
print(f"Will download {num_images} person images...")
print()

# 检查是否需要安装依赖
try:
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils
    print("✓ pycocotools is installed")
except ImportError:
    print("✗ pycocotools not found!")
    print()
    print("Installing pycocotools...")
    os.system("pip install pycocotools --user")
    print()
    try:
        from pycocotools.coco import COCO
        from pycocotools import mask as maskUtils
        print("✓ pycocotools installed successfully")
    except ImportError:
        print()
        print("=" * 70)
        print("ERROR: Failed to install pycocotools")
        print("=" * 70)
        print()
        print("Please install manually:")
        print("  pip install pycocotools")
        print()
        print("Or on Windows:")
        print("  pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI")
        print()
        input("Press Enter to exit...")
        sys.exit(1)

print()

# 下载 COCO 标注文件
ann_dir = os.path.join(dataset_dir, "annotations")
os.makedirs(ann_dir, exist_ok=True)

ann_file = os.path.join(ann_dir, "instances_train2017.json")
ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

if not os.path.exists(ann_file):
    print("Downloading COCO annotations...")
    print("This is a ~250MB file, please wait...")
    
    zip_path = os.path.join(dataset_dir, "annotations.zip")
    try:
        urllib.request.urlretrieve(ann_url, zip_path)
        print("✓ Annotations downloaded")
        
        # 解压
        print("Extracting annotations...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("✓ Annotations extracted")
        
        # 删除 zip 文件
        os.remove(zip_path)
        
    except Exception as e:
        print(f"✗ Error downloading annotations: {e}")
        print()
        print("Please download manually:")
        print("  http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        print(f"  Extract to: {dataset_dir}")
        input("Press Enter to exit...")
        sys.exit(1)
else:
    print("✓ Annotations already exist")

print()

# 加载 COCO API
print("Loading COCO API...")
coco = COCO(ann_file)

# 获取人物类别
cat_ids = coco.getCatIds(catNms=['person'])
print(f"✓ Found person category: {cat_ids}")

# 获取人物图片 ID
img_ids = coco.getImgIds(catIds=cat_ids)
print(f"✓ Found {len(img_ids)} images with person category")

# 限制数量
img_ids = img_ids[:num_images]
print(f"  Will download: {num_images} images")
print()

# 下载图片和生成掩码
print("Downloading images and generating masks...")
print("This may take a while, please wait...")
print()

import numpy as np
from PIL import Image

success_count = 0
for i, img_id in enumerate(img_ids):
    try:
        # 获取图片信息
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        img_file = img_info['file_name']
        
        # 下载图片
        img_path = os.path.join(images_dir, f"{i:05d}.jpg")
        urllib.request.urlretrieve(img_url, img_path)
        
        # 获取标注
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        
        # 创建掩码
        height = img_info['height']
        width = img_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for ann in anns:
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], list):
                    # 多边形格式
                    rles = maskUtils.frPyObjects(ann['segmentation'], height, width)
                    m = maskUtils.decode(rles)
                    if len(m.shape) == 3:
                        m = m.max(axis=2)
                    mask = np.maximum(mask, m * 255)
                elif 'counts' in ann['segmentation']:
                    # RLE 格式
                    m = maskUtils.decode(ann['segmentation'])
                    mask = np.maximum(mask, m * 255)
        
        # 保存掩码
        mask_path = os.path.join(masks_dir, f"{i:05d}.png")
        Image.fromarray(mask).save(mask_path)
        
        success_count += 1
        if (i + 1) % 10 == 0:
            print(f"  Downloaded {i + 1}/{num_images} images")
            
    except Exception as e:
        print(f"  ✗ Error with image {i}: {e}")
        continue

print()
print("=" * 70)
print(f"✓ Downloaded {success_count} images successfully!")
print("=" * 70)
print()
print(f"Images saved to: {images_dir}")
print(f"Masks saved to: {masks_dir}")
print()
print("Next step: Train U-Net with this dataset")
print("Run: python train_coco.py")
print()
input("Press Enter to exit...")
