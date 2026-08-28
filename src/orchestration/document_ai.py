import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.preprocessing.pipeline import to_grayscale, apply_clahe, otsu_threshold
from src.ocr.ocr_engine import extract_text_and_boxes
from src.extraction.extractor import extract_information

class Document_AI_Pipeline:
    def __init__(self, yolo_path, classifier_path):
        self.yolo = YOLO(yolo_path)
        self.device = torch.device("cuda")
        check_point = torch.load(classifier_path, map_location=self.device)
        self.class_names = check_point['class_names']

        self.classifier = models.resnet18()
        num_ftrs = self.classifier.fc.in_features
        self.classifier.fc = torch.nn.Linear(num_ftrs, len(self.class_names))
        self.classifier.load_state_dict(check_point['model_state_dict'])
        self.classifier.to(self.device)
        self.classifier.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("Pipeline loaded successfully")

    def process_image(self, original_img):
        gray = to_grayscale(original_img)
        clahe_img = apply_clahe(gray)
        preprocessed_img = otsu_threshold(clahe_img)
        ocr_tokens = extract_text_and_boxes(preprocessed_img)
        full_text = [token['text'] for token in ocr_tokens]
        pil_img = Image.fromarray(original_img)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.classifier(input_tensor)
            _, predicted = torch.max(outputs, 1)
            doc_type = self.class_names[predicted.item()]

        extracted_json = extract_information(doc_type, full_text)

        return {
            "document_type": doc_type,
            "extracted_fields": extracted_json,
            "word_count": len(full_text),
        }
