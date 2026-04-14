from src.data_loader import load_data
from src.model import build_model
from src.utils import plot_accuracy, plot_confusion_matrix
import os

def train_model():

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    data_dir = "data/"

    train_data, val_data = load_data(data_dir)

    model = build_model(num_classes=2)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=5
    )

    model.save("models/model.h5")

    plot_accuracy(history)
    plot_confusion_matrix(model, val_data)

    print("✅ Model trained & saved successfully!")