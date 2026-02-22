import os
import torch
import torch.nn as nn
from django.apps import AppConfig
from django.conf import settings
from torchvision import models
from ultralytics import YOLO
from models.labels import labels


class ClassifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'classifier'

    # We'll use these to store the actual objects once loaded
    _detector = None
    _classifier = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_names = labels

    @classmethod
    def get_detector(cls):
        if cls._detector is None:
            print("--- Loading YOLO Detector ---")
            path = os.path.join(settings.BASE_DIR, 'models', 'yolov8_animeface.pt')
            cls._detector = YOLO(path)
        return cls._detector

    @classmethod
    def get_classifier(cls):
        if cls._classifier is None:
            print("--- Loading ResNet Classifier ---")
            path = os.path.join(settings.BASE_DIR, 'models', 'anime_classifier_final.pth')

            model = models.resnet18(weights=None)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, len(cls.class_names))
            model.load_state_dict(torch.load(path, map_location=cls.device))
            model.eval()
            cls._classifier = model.to(cls.device)
        return cls._classifier