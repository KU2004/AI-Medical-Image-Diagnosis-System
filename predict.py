import tensorflow as tf
import numpy as np
import cv2
from src.preprocessing import preprocess_image  # ✅ reuse preprocessing

class_names = ["Normal", "Pneumonia"]

def predict_image(img_path):
    
    # Load trained model
    model = tf.keras.models.load_model("models/model.h5")

    # Read image
    img = cv2.imread(img_path)

    # ❌ If image not found
    if img is None:
        print("❌ Error: Image not found. Check file path.")
        return

    # ✅ Convert BGR → RGB (OpenCV loads in BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ✅ Preprocess image (resize, normalize, expand dims)
    img = preprocess_image(img)

    # Prediction
    prediction = model.predict(img)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    # Output
    print("\n🧠 Prediction Result")
    print("----------------------")
    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.2f}%")