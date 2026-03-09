
import sys
print("Python path:", sys.path)
print()

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except Exception as e:
    print(f"NumPy import failed: {e}")

try:
    import skimage
    print(f"scikit-image version: {skimage.__version__}")
except Exception as e:
    print(f"scikit-image import failed: {e}")
