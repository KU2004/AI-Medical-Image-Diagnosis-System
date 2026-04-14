from src.train import train_model
from src.predict import predict_image

if __name__ == "__main__":

    print("\nAI Medical Image System")
    print("1. Train Model")
    print("2. Predict Image")

    choice = input("Enter choice: ")

    if choice == "1":
        train_model()

    elif choice == "2":
        path = input("Enter image path: ")
        predict_image(path)