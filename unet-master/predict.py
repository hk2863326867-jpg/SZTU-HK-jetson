
import sys
import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
from model import unet

print("=" * 50)
print("U-Net Prediction - Loading trained model")
print("=" * 50)
print()

# Load the trained model
print("[1/4] Loading model...")
model = unet()
model.load_weights('unet_membrane.keras')
print("✓ Model loaded successfully!")
print()

print("[2/4] Preparing test images...")
test_path = "data/membrane/test"
num_image = 30
target_size = (256, 256)

results = []
for i in range(num_image):
    img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=True)
    img = img / 255
    img = trans.resize(img, target_size)
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img, (1,) + img.shape)
    results.append(img)

print(f"✓ Loaded {num_image} test images")
print()

print("[3/4] Making predictions...")
predictions = []
for i, img in enumerate(results):
    pred = model.predict(img, verbose=0)
    predictions.append(pred[0])
    print(f"  Processed {i+1}/{num_image}")

print("✓ Predictions complete!")
print()

print("[4/4] Saving results...")
save_path = "data/membrane/test"
for i, item in enumerate(predictions):
    img = item[:, :, 0]
    img = (img * 255).astype(np.uint8)
    io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)
    print(f"  Saved {i}_predict.png")

print()
print("=" * 50)
print("✓ ALL DONE!")
print(f"✓ Predictions saved to: {save_path}")
print("=" * 50)
print()
print("Now you can view the results in data/membrane/test/")
print("- 0.png to 29.png: Original test images")
print("- 0_predict.png to 29_predict.png: U-Net predictions")
