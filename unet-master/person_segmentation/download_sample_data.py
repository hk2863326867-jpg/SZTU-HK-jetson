
import os
import sys

print("=" * 70)
print("     Person Segmentation - Auto Create Sample Data")
print("=" * 70)
print()

# Create directories
dirs = [
    'dataset/train/image',
    'dataset/train/label',
    'dataset/test'
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    print(f"Created: {d}")

print()
print("=" * 70)
print("Creating sample data...")
print("=" * 70)
print()

# Use skimage which is already installed
import numpy as np
import skimage.io as io
from skimage.draw import ellipse, rectangle

print("[1/3] Creating sample training images...")
print()

# Create 10 sample training images with simple person silhouettes
for i in range(10):
    # Create a random background
    img = np.ones((256, 256, 3), dtype=np.uint8) * 150
    
    # Add some random noise to background
    img = img + np.random.randint(-30, 30, (256, 256, 3), dtype=np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Random position offset
    offset_x = np.random.randint(-15, 15)
    offset_y = np.random.randint(-10, 10)
    
    # Draw a simple person silhouette
    # Head (ellipse)
    head_center = (60 + offset_y, 128 + offset_x)
    head_shape = (25, 20)
    rr, cc = ellipse(head_center[0], head_center[1], head_shape[0], head_shape[1])
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    img[rr[valid], cc[valid]] = [200, 150, 100]
    
    # Body (rectangle)
    body_start = (90, 98 + offset_x)
    body_end = (180, 158 + offset_x)
    rr, cc = rectangle(start=body_start, end=body_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    img[rr[valid], cc[valid]] = [100, 100, 200]
    
    # Left leg
    leg1_start = (180, 103 + offset_x)
    leg1_end = (250, 123 + offset_x)
    rr, cc = rectangle(start=leg1_start, end=leg1_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    img[rr[valid], cc[valid]] = [50, 50, 150]
    
    # Right leg
    leg2_start = (180, 133 + offset_x)
    leg2_end = (250, 153 + offset_x)
    rr, cc = rectangle(start=leg2_start, end=leg2_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    img[rr[valid], cc[valid]] = [50, 50, 150]
    
    # Left arm
    arm1_start = (100, 78 + offset_x)
    arm1_end = (150, 98 + offset_x)
    rr, cc = rectangle(start=arm1_start, end=arm1_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    img[rr[valid], cc[valid]] = [100, 100, 200]
    
    # Right arm
    arm2_start = (100, 158 + offset_x)
    arm2_end = (150, 178 + offset_x)
    rr, cc = rectangle(start=arm2_start, end=arm2_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    img[rr[valid], cc[valid]] = [100, 100, 200]
    
    # Save image
    io.imsave(f'dataset/train/image/{i}.png', img)
    print(f"  Created training image {i}.png")

print()
print("[2/3] Creating sample training labels (masks)...")
print()

# Create corresponding masks
for i in range(10):
    # Create a black mask
    mask = np.zeros((256, 256), dtype=np.uint8)
    
    # Random position offset (same as image)
    offset_x = np.random.randint(-15, 15)
    offset_y = np.random.randint(-10, 10)
    
    # Draw white person silhouette
    # Head
    head_center = (60 + offset_y, 128 + offset_x)
    head_shape = (25, 20)
    rr, cc = ellipse(head_center[0], head_center[1], head_shape[0], head_shape[1])
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    mask[rr[valid], cc[valid]] = 255
    
    # Body
    body_start = (90, 98 + offset_x)
    body_end = (180, 158 + offset_x)
    rr, cc = rectangle(start=body_start, end=body_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    mask[rr[valid], cc[valid]] = 255
    
    # Left leg
    leg1_start = (180, 103 + offset_x)
    leg1_end = (250, 123 + offset_x)
    rr, cc = rectangle(start=leg1_start, end=leg1_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    mask[rr[valid], cc[valid]] = 255
    
    # Right leg
    leg2_start = (180, 133 + offset_x)
    leg2_end = (250, 153 + offset_x)
    rr, cc = rectangle(start=leg2_start, end=leg2_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    mask[rr[valid], cc[valid]] = 255
    
    # Left arm
    arm1_start = (100, 78 + offset_x)
    arm1_end = (150, 98 + offset_x)
    rr, cc = rectangle(start=arm1_start, end=arm1_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    mask[rr[valid], cc[valid]] = 255
    
    # Right arm
    arm2_start = (100, 158 + offset_x)
    arm2_end = (150, 178 + offset_x)
    rr, cc = rectangle(start=arm2_start, end=arm2_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    mask[rr[valid], cc[valid]] = 255
    
    # Save mask
    io.imsave(f'dataset/train/label/{i}.png', mask)
    print(f"  Created training label {i}.png")

print()
print("[3/3] Creating sample test images...")
print()

# Create 5 test images
for i in range(5):
    img = np.ones((256, 256, 3), dtype=np.uint8) * 150
    img = img + np.random.randint(-30, 30, (256, 256, 3), dtype=np.int16)
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    offset_x = np.random.randint(-15, 15)
    offset_y = np.random.randint(-10, 10)
    
    # Head
    head_center = (60 + offset_y, 128 + offset_x)
    head_shape = (25, 20)
    rr, cc = ellipse(head_center[0], head_center[1], head_shape[0], head_shape[1])
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    img[rr[valid], cc[valid]] = [200, 150, 100]
    
    # Body
    body_start = (90, 98 + offset_x)
    body_end = (180, 158 + offset_x)
    rr, cc = rectangle(start=body_start, end=body_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    img[rr[valid], cc[valid]] = [100, 100, 200]
    
    # Left leg
    leg1_start = (180, 103 + offset_x)
    leg1_end = (250, 123 + offset_x)
    rr, cc = rectangle(start=leg1_start, end=leg1_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    img[rr[valid], cc[valid]] = [50, 50, 150]
    
    # Right leg
    leg2_start = (180, 133 + offset_x)
    leg2_end = (250, 153 + offset_x)
    rr, cc = rectangle(start=leg2_start, end=leg2_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    img[rr[valid], cc[valid]] = [50, 50, 150]
    
    # Left arm
    arm1_start = (100, 78 + offset_x)
    arm1_end = (150, 98 + offset_x)
    rr, cc = rectangle(start=arm1_start, end=arm1_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    img[rr[valid], cc[valid]] = [100, 100, 200]
    
    # Right arm
    arm2_start = (100, 158 + offset_x)
    arm2_end = (150, 178 + offset_x)
    rr, cc = rectangle(start=arm2_start, end=arm2_end)
    valid = (rr >= 0) & (rr < 256) & (cc >= 0) & (cc < 256)
    img[rr[valid], cc[valid]] = [100, 100, 200]
    
    io.imsave(f'dataset/test/{i}.png', img)
    print(f"  Created test image {i}.png")

print()
print("=" * 70)
print("✓ Sample data creation complete!")
print("=" * 70)
print()
print("Summary:")
print(f"  - Training images: 10 (dataset/train/image/)")
print(f"  - Training labels: 10 (dataset/train/label/)")
print(f"  - Test images: 5 (dataset/test/)")
print()
print("Next steps:")
print("  1. Run: python train_person.py")
print("  2. Wait for training to complete")
print("  3. Run: python predict_person.py")
print("  4. Check results in dataset/test/")
print()
print("Note: This is a simple demo dataset with synthetic images.")
print("For better results with real person photos, download a")
print("larger dataset like Supervisely Person or Pascal VOC.")
print()
