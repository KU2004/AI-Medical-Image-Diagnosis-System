import cv2
import numpy as np

# -----------------------------
# FUNCTION: PREPROCESS IMAGE
# -----------------------------
def preprocess_image(image, target_size=(224, 224)):
    """
    Resize, normalize and prepare image for model
    """

    # Convert PIL → NumPy if needed
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # ✅ FIX 1: Handle grayscale images (2D → 3 channels)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # ✅ FIX 2: Handle single channel images (224,224,1)
    if len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize image
    image = cv2.resize(image, target_size)

    # Normalize pixel values (0–255 → 0–1)
    image = image.astype("float32") / 255.0

    # Expand dimensions (1, 224, 224, 3)
    image = np.expand_dims(image, axis=0)

    return image