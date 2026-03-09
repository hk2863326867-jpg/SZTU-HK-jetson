
import sys
import os
import numpy as np
import skimage.io as io
import skimage.transform as trans
from model import unet

print("=" * 50)
print("Saving U-Net Prediction Results")
print("=" * 50)
print()

print("[1/3] Loading model...")
model = unet()
model.load_weights('unet_membrane.keras')
print("✓ Model loaded")
print()

print("[2/3] Making predictions (quick)...")
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

predictions = []
for i, img in enumerate(results):
    pred = model.predict(img, verbose=0)
    predictions.append(pred[0])
print("✓ Predictions done")
print()

print("[3/3] Saving results...")
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
print("Now you can view:")
print("  - data/membrane/test/0.png  (original)")
print("  - data/membrane/test/0_predict.png  (prediction)")
