
import sys
import os
import numpy as np
from model import unet
from data import trainGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import skimage.io as io
import skimage.transform as trans

print("=" * 60)
print("U-Net Full Pipeline - Train + Predict + Save")
print("=" * 60)
print()

# ==========================================
# STEP 1: TRAINING
# ==========================================
print("[1/4] Setting up data generator...")
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)
print("✓ Data generator ready")
print()

print("[2/4] Creating and training model...")
model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.keras', monitor='loss', verbose=1, save_best_only=True)
print("Starting training (this will take a while)...")
print()
model.fit(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])
print()
print("✓ Training complete! Model saved as unet_membrane.keras")
print()

# ==========================================
# STEP 2: PREDICTION
# ==========================================
print("[3/4] Making predictions...")
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
    print(f"  Predicted {i+1}/{num_image}")
print("✓ Predictions complete")
print()

# ==========================================
# STEP 3: SAVE RESULTS
# ==========================================
print("[4/4] Saving prediction results...")
save_path = "data/membrane/test"
for i, item in enumerate(predictions):
    img = item[:, :, 0]
    img = (img * 255).astype(np.uint8)
    io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)
    print(f"  Saved {i}_predict.png")

print()
print("=" * 60)
print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
print("=" * 60)
print()
print("Results saved to: data/membrane/test/")
print("  - Original images: 0.png - 29.png")
print("  - Predictions: 0_predict.png - 29_predict.png")
print()
