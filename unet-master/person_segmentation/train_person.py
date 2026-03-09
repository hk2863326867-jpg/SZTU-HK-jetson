
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from model_rgb import unet_rgb
from data_rgb import trainGenerator_rgb, testGenerator_rgb, saveResult_rgb
from tensorflow.keras.callbacks import ModelCheckpoint
import skimage.io as io
import skimage.transform as trans

print("=" * 70)
print("           Person Segmentation - Full Training Pipeline")
print("=" * 70)
print()

# ==========================================
# STEP 1: SETUP DATA GENERATOR
# ==========================================
print("[1/4] Setting up data generator...")
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

train_path = 'dataset/train'
myGene = trainGenerator_rgb(2, train_path, 'image', 'label', data_gen_args, save_to_dir=None)
print("✓ Data generator ready")
print(f"  Training data path: {train_path}")
print()

# ==========================================
# STEP 2: CREATE AND TRAIN MODEL
# ==========================================
print("[2/4] Creating and training model...")
model = unet_rgb()
model_checkpoint = ModelCheckpoint('person_unet.keras', monitor='loss', verbose=1, save_best_only=True)

print("Starting training...")
print("  Batch size: 2")
print("  Steps per epoch: 300")
print("  Epochs: 1")
print("  Input size: 256x256x3 (RGB)")
print()
print("This will take some time, please wait...")
print()

model.fit(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])
print()
print("✓ Training complete! Model saved as person_unet.keras")
print()

# ==========================================
# STEP 3: PREDICTION
# ==========================================
print("[3/4] Making predictions...")
test_path = "dataset/test"

if not os.path.exists(test_path):
    print(f"Warning: Test directory '{test_path}' not found!")
    print("Creating test directory...")
    os.makedirs(test_path, exist_ok=True)
    print("Please add test images to dataset/test/ and run predict_person.py")
else:
    num_image = len([f for f in os.listdir(test_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if num_image == 0:
        print(f"Warning: No test images found in {test_path}")
        print("Please add test images and run predict_person.py")
    else:
        print(f"Found {num_image} test images")
        
        results = []
        for i, img in enumerate(testGenerator_rgb(test_path, num_image=num_image)):
            pred = model.predict(img, verbose=0)
            results.append(pred[0])
            print(f"  Predicted {i+1}/{num_image}")
        
        print("✓ Predictions complete")
        print()
        
        # ==========================================
        # STEP 4: SAVE RESULTS
        # ==========================================
        print("[4/4] Saving prediction results...")
        for i, item in enumerate(results):
            img = item[:, :, 0]
            img = (img * 255).astype(np.uint8)
            io.imsave(os.path.join(test_path, "%d_predict.png" % i), img)
            print(f"  Saved {i}_predict.png")
        
        print()
        print("=" * 70)
        print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print(f"Results saved to: {test_path}")
        print()

print()
print("=" * 70)
print("Training complete!")
print("=" * 70)
print()
print("Next steps:")
print("  1. Add test images to dataset/test/")
print("  2. Run predict_person.py to generate predictions")
print()
