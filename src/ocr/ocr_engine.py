# src/ocr/ocr_engine.py
import pytesseract
import cv2
import os

# If you are on Windows, uncomment the line below and set your tesseract path:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_and_boxes(img):
    """
    Extracts text, bounding boxes, and confidence scores from an image.
    Returns a list of dictionaries matching the roadmap's required format.
    """
    # Run Tesseract with bounding box output
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    results = []
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        
        # Only keep tokens that have actual text and a positive confidence score
        if text and conf > 0:
            # Tesseract returns [x, y, width, height]. We want [x1, y1, x2, y2]
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            bbox = [x, y, x + w, y + h]
            
            results.append({
                "text": text,
                "bbox": bbox,
                "confidence": conf / 100.0  # Normalize to 0.0 - 1.0
            })
            
    return results