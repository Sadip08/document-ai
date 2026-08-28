import os
import sys
import numpy as np
import cv2
import fitz
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.orchestration.document_ai import Document_AI_Pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

YOLO_PATH = "E:\\document-ai\\models\\layout_detection_exp\\weights\\best.pt"
CLASSIFIER_PATH = "E:\\document-ai\\models\\doc_classifier_resnet18.pth"

try:
    pipeline = Document_AI_Pipeline(YOLO_PATH, CLASSIFIER_PATH)
except Exception as e:
    print(f"failed to initialize pipeline: {e}")
    pipeline = None

@app.get("/health")
def health_check():
    return {'status': "healthy", 'pipeline_loaded' : pipeline is not None}

@app.post("/predict")
async def predict_document(file: UploadFile = File(...)):
    if not pipeline:
        raise HTTPException(status_code=500, detail= "Pipeline not initialized")

    #Read upload file into memory
    contents = await file.read()
    file_ext = file.filename.split('.')[-1].lower()

    try:
        if file_ext == "pdf":
            # Render first page of PDF
            with fitz.open(stream=contents, filetype="pdf") as doc:
                pix = doc[0].get_pixmap(dpi=300)

            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)            

        else:
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        result = pipeline.process_image(img)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")