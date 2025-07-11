import os
from ultralytics import YOLO

# --- Configuration for Training ---
# Path to your data.yaml file (ABSOLUTE PATH MODIFIED)
DATA_YAML_PATH = "C:/Users/KIIT0001/Desktop/ML & IOT In Sub-Stations/FootwearSafetyProject/data.yaml"



# Pre-trained YOLOv8 model to use as a base for fine-tuning.
MODEL_FOR_TRAINING = 'yolov8n.pt'

# --- Training Parameters for YOUR Custom Shoe Detector ---
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
# DEVICE = 0

def train_custom_shoe_detector():
    """
    Initializes and trains your custom 'shoes' YOLOv8 model using your annotated dataset.
    """
    print(f"\n--- Starting Custom Shoe Detector Training with {MODEL_FOR_TRAINING} ---")

    if not os.path.exists(DATA_YAML_PATH):
        print(f"Error: '{DATA_YAML_PATH}' not found. Please ensure it's in the project root.")
        print("Training cannot proceed without data.yaml.")
        return

    model = YOLO(MODEL_FOR_TRAINING)

    try:
        results = model.train(
            data=DATA_YAML_PATH,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            # device=DEVICE,
            cache=True,
            val=True,
        )
        print("\n--- Custom Shoe Detector Training completed successfully! ---")
        save_dir = model.trainer.save_dir
        print(f"Training results and model saved to: {save_dir}")
        print(f"Your best trained model weights are at: {save_dir}/weights/best.pt")
    except Exception as e:
        print(f"An error occurred during shoe detector training: {e}")
        print("Please ensure your dataset is properly annotated and organized under 'dataset/' folder.")
        print("Also check your 'data.yaml' paths and content.")

if __name__ == "__main__":
    train_custom_shoe_detector()