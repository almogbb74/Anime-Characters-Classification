import os
import cv2
import numpy as np
import torch
import uuid

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from torchvision import transforms
from config import settings
from .apps import ClassifierConfig
from django.shortcuts import render
from PIL import Image
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


def index(request):
    return render(request, 'index.html')


@csrf_exempt
def classify_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        file = request.FILES['image']

        # Handle File Naming & Saving
        ext = os.path.splitext(file.name)[1]
        unique_filename = f"{uuid.uuid4()}{ext}"

        # Save to the media directory defined in your settings
        file_path = default_storage.save(unique_filename, ContentFile(file.read()))
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)

        try:
            # Load Image with OpenCV
            img = cv2.imread(full_path)

            if img is None:
                default_storage.delete(file_path)
                return JsonResponse({'name': 'Invalid image format'}, status=400)

            detector = ClassifierConfig.get_detector()
            results = detector(img, verbose=False, device=ClassifierConfig.device)

            if len(results[0].boxes) == 0:
                default_storage.delete(file_path)
                return JsonResponse({'name': 'No face detected'})

            # Logic to find the largest detected face
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_box = boxes[np.argmax(areas)]

            # Crop Face
            x1, y1, x2, y2 = map(int, best_box)
            face_crop = img[y1:y2, x1:x2]

            # Preprocessing & Classification (ResNet)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Convert BGR (OpenCV) to RGB (PIL/Torch)
            img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_rgb)
            image_tensor = transform(pil_image).unsqueeze(0).to(ClassifierConfig.device)

            with torch.no_grad():
                classifier = ClassifierConfig.get_classifier()
                with torch.no_grad():
                    outputs = classifier(image_tensor)
                _, top_class_idx = torch.max(outputs, 1)

                # Get name from labels list loaded in apps.py
                predicted_name = ClassifierConfig.class_names[top_class_idx.item()]

            # Clean up and Return Result
            default_storage.delete(file_path)
            return JsonResponse({'name': predicted_name.replace('_', ' ')})

        except Exception as e:
            # Always attempt to delete the temp file if a crash occurs
            if default_storage.exists(file_path):
                default_storage.delete(file_path)
            return JsonResponse({'name': f'Error: {str(e)}'}, status=500)

    return JsonResponse({'name': 'Invalid Request'}, status=400)