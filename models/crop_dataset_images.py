import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# Ensure yolov8_animeface.pt model is installed.
# if not, install via:
# wget -O yolov8_animeface.pt "https://huggingface.co/Fuyucchi/yolov8_animeface/resolve/main/yolov8x6_animeface.pt"
# and raw_dataset is NOT included in the project.

RAW_DATASET = 'raw_dataset'
OUTPUT_DATASET = 'output_dataset'
MODEL_NAME = 'yolov8_animeface.pt'


def check_gpu():
    print('\n=== SYSTEM CHECK ===')
    print(torch.__version__)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f'GPU Detected: {gpu_name}')
        print('Running on GPU!')
        return 'cuda'
    else:
        print('GPU NOT detected. Running on CPU.')
        return 'cpu'


def crop_face(image, box, pad=0.2):
    # Crops the face from the image with padding.
    x1, y1, x2, y2 = map(int, box)

    # Calculate padding
    h = y2 - y1
    w = x2 - x1
    x1p = max(0, int(x1 - w * pad))
    y1p = max(0, int(y1 - h * pad))
    x2p = min(image.shape[1], int(x2 + w * pad))
    y2p = min(image.shape[0], int(y2 + h * pad))

    face_crop = image[y1p:y2p, x1p:x2p]
    return face_crop


def rename_images():
    print(f'Renaming files in: {OUTPUT_DATASET}')

    # Get all character folders
    characters = sorted([d for d in os.listdir(OUTPUT_DATASET) if os.path.isdir(os.path.join(OUTPUT_DATASET, d))])

    for character in tqdm(characters, desc="Total Progress"):
        folder_path = os.path.join(OUTPUT_DATASET, character)

        # Get all files, sorted to ensure consistent order
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])

        # Create a temp list to avoid issues modifying the directory while reading it
        for idx, filename in enumerate(tqdm(files, desc=f"Renaming {character}", leave=False), start=1):

            # Get the file extension (e.g., .png or .jpg)
            file_ext = os.path.splitext(filename)[1]

            # FORMAT: CharacterName_001.png
            # Using :04d gives you 3 digits: 001, 002... (good for up to 999 images)
            new_name = f"{character}_{idx:03d}{file_ext}"

            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_name)

            # Skip if file is already renamed to avoid errors
            if old_path == new_path:
                continue

            os.rename(old_path, new_path)


def main():
    # Setup Device
    device = check_gpu()
    # # Load Model
    print(f'Loading {MODEL_NAME}...')
    try:
        model = YOLO(MODEL_NAME)
        model.to(device)  # Force model to GPU
    except Exception as e:
        print(f'Error loading model: {e}')
        return

    # Prepare Output Directory
    os.makedirs(OUTPUT_DATASET, exist_ok=True)

    # Get list of characters (subdirectories)
    if not os.path.exists(RAW_DATASET):
        print(f"Error: The input path '{RAW_DATASET}' does not exist.")
        return

    characters = sorted([d for d in os.listdir(RAW_DATASET) if os.path.isdir(os.path.join(RAW_DATASET, d))])

    print(f'\nfound {len(characters)} character folders. Starting processing...')

    # Process loop
    total_images_processed = 0

    for character in characters:
        input_dir = os.path.join(RAW_DATASET, character)
        output_dir = os.path.join(OUTPUT_DATASET, character)
        os.makedirs(output_dir, exist_ok=True)
        img_counter = 1

        # Get list of images
        images = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]

        # We use tqdm here to show a progress bar for each character folder
        # desc=character shows the current character name in the bar
        for img_name in tqdm(images, desc=f'Processing {character}', unit='img'):

            img_path = os.path.join(input_dir, img_name)

            # Read Image
            img = cv2.imread(img_path)
            if img is None:
                continue

            results = model(img, verbose=False, device=device)

            # Check if faces detected
            if len(results[0].boxes) == 0:
                continue

            # Find the largest face
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_box = boxes[np.argmax(areas)]

            # Crop and Save
            cropped_face = crop_face(img, best_box)

            new_filename = f"{character}_{img_counter:03d}.png"  # Rename Images during saving

            out_path = os.path.join(output_dir, new_filename)
            cv2.imwrite(out_path, cropped_face)  # Save cropped image
            img_counter += 1
            total_images_processed += 1

    print(f'\nDone! Processed {total_images_processed} images.')
    print(f'Saved to: {OUTPUT_DATASET}')


if __name__ == '__main__':
    main()
