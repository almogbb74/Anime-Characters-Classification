import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# --- CONFIGURATION ---
MODEL_PATH = 'anime_classifier_final.pth'
DATA_DIR = 'output_dataset'
BATCH_SIZE = 32

# Please run split_dataset.py before you run this script to create the 'output_dataset' folder.

# ---------------------

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # Define Transforms (Must match training validation transforms)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Validation Data
    val_dir = os.path.join(DATA_DIR, 'val')
    if not os.path.exists(val_dir):
        print(f"Error: Validation folder not found at {val_dir}")
        return

    val_dataset = datasets.ImageFolder(val_dir, val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    class_names = val_dataset.classes
    print(f"Loaded {len(class_names)} classes.")

    # Load Model Structure
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    # Load Your Saved Weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Weights loaded successfully.")
    else:
        print("Error: Model file not found.")
        return

    model.to(device)
    model.eval()  # Set to evaluation mode

    # Run Evaluation
    running_corrects = 0
    total_samples = 0

    print("\nStarting evaluation...")

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Update overall stats
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

    # Print Results
    overall_acc = running_corrects / total_samples * 100
    print("-" * 30)
    print(f"ACCURACY: {overall_acc:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    evaluate_model()