"""
下载预训练的人物分割模型
使用 TensorFlow Hub 的预训练模型
"""
import os
import sys
import urllib.request
import zipfile

print("=" * 70)
print("     Downloading Pre-trained Person Segmentation Model")
print("=" * 70)
print()

# 创建模型目录
model_dir = "pretrained_model"
os.makedirs(model_dir, exist_ok=True)

print("[1/2] Checking dependencies...")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
except ImportError:
    print("✗ TensorFlow not found!")
    print("Installing TensorFlow...")
    os.system("pip install tensorflow --user")
    import tensorflow as tf

try:
    import tensorflow_hub as hub
    print("✓ TensorFlow Hub installed")
except ImportError:
    print("✗ TensorFlow Hub not found!")
    print("Installing TensorFlow Hub...")
    os.system("pip install tensorflow-hub --user")
    import tensorflow_hub as hub

print()
print("[2/2] Loading pre-trained model from TensorFlow Hub...")
print("This may take a few minutes on first run...")
print()

# 使用 TensorFlow Hub 的预训练人物分割模型
# 这是一个基于 DeepLab v3 的模型，在 COCO 数据集上训练
model_url = "https://tfhub.dev/tensorflow/deeplabv3/1"

try:
    print("Downloading model (this may take 5-10 minutes)...")
    model = hub.load(model_url)
    print("✓ Model loaded successfully!")
    
    # 保存模型
    print("Saving model for future use...")
    tf.saved_model.save(model, os.path.join(model_dir, "deeplabv3"))
    print(f"✓ Model saved to {model_dir}/deeplabv3")
    
    print()
    print("=" * 70)
    print("✓ Pre-trained model ready!")
    print("=" * 70)
    print()
    print("Next step: Run predict_pretrained.py to test with your photos")
    
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print()
    print("Alternative: Using OpenCV DNN module with pre-trained model...")
    print()
    
    # 备选方案：使用 OpenCV 的 DNN 模块
    try:
        import cv2
        print("✓ OpenCV found")
        
        # 下载 MobileNet 分割模型
        model_files = {
            "model.pb": "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/frozen_inference_graph.pb",
            "model.pbtxt": "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
        }
        
        print("Downloading OpenCV DNN model files...")
        for filename, url in model_files.items():
            filepath = os.path.join(model_dir, filename)
            if not os.path.exists(filepath):
                print(f"  Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"  ✓ {filename} downloaded")
            else:
                print(f"  ✓ {filename} already exists")
        
        print()
        print("=" * 70)
        print("✓ OpenCV model ready!")
        print("=" * 70)
        
    except Exception as e2:
        print(f"✗ Alternative also failed: {e2}")
        print()
        print("Please check your internet connection and try again.")

print()
input("Press Enter to exit...")
