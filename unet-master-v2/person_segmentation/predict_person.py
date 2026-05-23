
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import skimage.io as io
import skimage.transform as trans
from model_rgb import unet_rgb

print("=" * 70)
print("           Person Segmentation - Quick Prediction")
print("=" * 70)
print()

# ==========================================
# STEP 1: LOAD MODEL
# ==========================================
print("[1/3] Loading trained model...")

if not os.path.exists('person_unet.keras'):
    print("ERROR: person_unet.keras not found!")
    print("Please run train_person.py first to train the model.")
    sys.exit(1)

model = unet_rgb()
model.load_weights('person_unet.keras')
print("✓ Model loaded from person_unet.keras")
print()

# ==========================================
# STEP 2: PREDICTION
# ==========================================
print("[2/3] Making predictions...")
test_path = "dataset/test"

if not os.path.exists(test_path):
    print(f"ERROR: Test directory '{test_path}' not found!")
    sys.exit(1)

image_files = [f for f in os.listdir(test_path) 
               if f.endswith(('.png', '.jpg', '.jpeg')) and '_predict' not in f]

if len(image_files) == 0:
    print(f"ERROR: No test images found in {test_path}")
    print("Please add test images (0.png, 1.png, ... or 0.jpg, 1.jpg, ...)")
    sys.exit(1)

num_image = len(image_files)
print(f"Found {num_image} test images")
print()

results = []
for i in range(num_image):
    img_path = os.path.join(test_path, "%d.png" % i)
    if not os.path.exists(img_path):
        img_path = os.path.join(test_path, "%d.jpg" % i)
    
    if not os.path.exists(img_path):
        print(f"Warning: Image {i} not found, skipping...")
        continue
    
    img = io.imread(img_path)
    
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=2)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    
    img = img / 255
    img = trans.resize(img, (256, 256))
    img = np.reshape(img, (1,) + img.shape)
    
    pred = model.predict(img, verbose=0)
    results.append((i, pred[0]))
    print(f"  Predicted {i+1}/{num_image}")

print("✓ Predictions complete")
print()

# ==========================================
# STEP 3: SAVE RESULTS
# ==========================================
print("[3/3] Saving results...")
for i, item in results:
    img = item[:, :, 0]
    img = (img * 255).astype(np.uint8)
    io.imsave(os.path.join(test_path, "%d_predict.png" % i), img)
    print(f"  Saved {i}_predict.png")

print()
print("=" * 70)
print("✓ ALL DONE!")
print("=" * 70)
print()
print(f"Results saved to: {test_path}")
print()
print("View the results:")
print(f"  - Original: {test_path}/0.png (or 0.jpg)")
print(f"  - Prediction: {test_path}/0_predict.png")
print()
