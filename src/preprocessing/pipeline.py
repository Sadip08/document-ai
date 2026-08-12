import cv2
import numpy as np
import fitz  # PyMuPDF

def render_pdf_page(pdf_path: str, page_num: int = 0, dpi: int = 300) -> np.ndarray:
    """Converts a PDF page into a high-resolution RGB numpy array."""
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, pix.n)
    
    # Handle alpha channel if present
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    doc.close()
    return img

def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """Applies Gaussian or Median filtering."""
    if method == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 0)
    elif method == "median":
        return cv2.medianBlur(img, 3)
    return img

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: tuple = (8, 8)) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

def otsu_threshold(img: np.ndarray) -> np.ndarray:
    """Applies Otsu's binarization."""
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def preprocess_document(img: np.ndarray) -> dict:
    """Runs the full baseline preprocessing pipeline and returns all stages for visualization."""
    stages = {}
    
    stages["original"] = img
    gray = to_grayscale(img)
    stages["grayscale"] = gray
    
    denoised = denoise(gray, method="gaussian")
    stages["denoised"] = denoised
    
    clahe = apply_clahe(denoised)
    stages["clahe"] = clahe
    
    thresholded = otsu_threshold(clahe)
    stages["thresholded"] = thresholded
    
    return stages