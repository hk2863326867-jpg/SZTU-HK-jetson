"""
下载 Pascal VOC 2012 数据集中的人物分割数据
"""
import os
import sys
import urllib.request
import tarfile
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np

print("=" * 70)
print("     Downloading Pascal VOC 2012 Person Dataset")
print("=" * 70)
print()

# 创建目录
dataset_dir = "pascal_voc_person"
images_dir = os.path.join(dataset_dir, "images")
masks_dir = os.path.join(dataset_dir, "masks")

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

print("This will download person images from Pascal VOC 2012 dataset.")
print()
print("Dataset info:")
print("  - Source: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/")
print("  - Size: ~500MB")
print("  - Contains: Real photos with person segmentation masks")
print()

response = input("Continue? (y/n) [default=y]: ").strip().lower()
if response == 'n':
    print("Download cancelled.")
    sys.exit(0)

print()

# 下载链接
VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
tar_path = os.path.join(dataset_dir, "VOCtrainval_11-May-2012.tar")

# 下载数据集
if not os.path.exists(tar_path):
    print("Downloading Pascal VOC 2012 dataset...")
    print("This is a ~500MB file, please wait...")
    print()
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if percent % 10 == 0:
                print(f"  Downloaded: {percent}%", end='\r')
        
        urllib.request.urlretrieve(VOC_URL, tar_path, reporthook=progress_hook)
        print("\n✓ Download complete!")
        
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print()
        print("Please download manually from:")
        print("  http://host.robots.ox.ac.uk/pascal/VOC/voc2012/")
        print(f"  Save to: {tar_path}")
        input("Press Enter to exit...")
        sys.exit(1)
else:
    print("✓ Dataset file already exists")

print()

# 解压数据集
extract_dir = os.path.join(dataset_dir, "VOCdevkit")
if not os.path.exists(extract_dir):
    print("Extracting dataset...")
    try:
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(dataset_dir)
        print("✓ Extraction complete!")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
else:
    print("✓ Dataset already extracted")

print()

# 处理数据
print("Processing person images...")
print()

voc_root = os.path.join(extract_dir, "VOC2012")
image_dir = os.path.join(voc_root, "JPEGImages")
seg_dir = os.path.join(voc_root, "SegmentationClass")

# 获取所有分割图像
seg_images = [f for f in os.listdir(seg_dir) if f.endswith('.png')]
print(f"Found {len(seg_images)} segmented images")

# 人物类别在 Pascal VOC 中的 ID 是 15
PERSON_ID = 15

person_count = 0
for i, seg_file in enumerate(seg_images):
    try:
        # 读取分割掩码
        seg_path = os.path.join(seg_dir, seg_file)
        seg_img = Image.open(seg_path)
        seg_array = np.array(seg_img)
        
        # 检查是否包含人物
        if PERSON_ID in seg_array:
            # 获取对应的原始图像
            img_name = seg_file.replace('.png', '.jpg')
            img_path = os.path.join(image_dir, img_name)
            
            if os.path.exists(img_path):
                # 复制原始图像
                original_img = Image.open(img_path)
                original_img.save(os.path.join(images_dir, f"{person_count:05d}.jpg"))
                
                # 创建人物掩码（二值化）
                person_mask = (seg_array == PERSON_ID).astype(np.uint8) * 255
                mask_img = Image.fromarray(person_mask)
                mask_img.save(os.path.join(masks_dir, f"{person_count:05d}.png"))
                
                person_count += 1
                
                if person_count % 50 == 0:
                    print(f"  Processed {person_count} person images...")
    
    except Exception as e:
        print(f"  ✗ Error processing {seg_file}: {e}")
        continue

print()
print("=" * 70)
print(f"✓ Extracted {person_count} person images!")
print("=" * 70)
print()
print(f"Images saved to: {images_dir}")
print(f"Masks saved to: {masks_dir}")
print()

if person_count == 0:
    print("⚠ Warning: No person images found!")
    print("This might be due to dataset structure changes.")
else:
    print("Next step: Train U-Net with this dataset")
    print("Run: python train_pascal.py")

print()
input("Press Enter to exit...")
