import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import os
import copy


def main():
    # --- CONFIGURATION ---
    DATA_DIR = 'output_dataset'
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001

    # Please run split_dataset.py before you run this script to create the 'output_dataset' folder.

    # ---------------------

    # Setup Device (GPU is much faster)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define Transforms (Must match ResNet requirements)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet standard size
            transforms.RandomHorizontalFlip(),  # Augmentation: Flip left/right
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load Data
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f"Classes: {len(class_names)}")
    print(f"Training images: {dataset_sizes['train']}")
    print(f"Validation images: {dataset_sizes['val']}")

    # Load Pre-trained ResNet18
    print("Downloading ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # FREEZE the early layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the "Head"
    # model.fc is the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(class_names))  # Output = number of your characters

    model = model.to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    # We only optimize parameters of the final layer (model.fc)
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    # Training Loop
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save checkpoint immediately
                torch.save(model.state_dict(), 'anime_classifier_best.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save Final Model
    torch.save(model.state_dict(), 'anime_classifier_final.pth')
    print("Model saved as 'anime_classifier_final.pth'")


if __name__ == '__main__':
    main()