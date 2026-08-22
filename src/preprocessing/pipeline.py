import cv2
import numpy as np
import fitz  # PyMuPDF

def render_pdf_page(pdf_path: str, page_num: int = 0, dpi: int = 300) -> np.ndarray:
    """Converts a PDF page into a high-resolution RGB numpy array."""
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        # Convert to OpenCV's BGR format
        if pix.n == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """Applies Gaussian or Median filtering."""
    filters = {
        "gaussian": lambda x: cv2.GaussianBlur(x, (5, 5), 0),
        "median": lambda x: cv2.medianBlur(x, 3),
    }

    return filters.get(method, lambda x: x)(img)

def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, grid_size: tuple = (8, 8)) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)

def otsu_threshold(img: np.ndarray) -> np.ndarray:
    """Applies Otsu's binarization."""
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def preprocess_document(img: np.ndarray) -> dict[str, np.ndarray]:
    """Runs the full baseline preprocessing pipeline and returns all stages for visualization."""
    gray = to_grayscale(img)
    denoised = denoise(gray)
    clahe = apply_clahe(denoised)
    thresholded = otsu_threshold(clahe)

    return {
        "original": img,
        "grayscale": gray,
        "denoised": denoised,
        "clahe": clahe,
        "thresholded": thresholded,
    }

MORPH_KERNEL = np.ones((2, 2), np.uint8)

def morphological_cleanup(img: np.ndarray) -> np.ndarray:
    """Applies morphological opening to remove small noise dots."""
    # Opening is erosion followed by dilation. Good for removing noise.
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, MORPH_KERNEL)

def deskew(img: np.ndarray) -> np.ndarray:
    """Detects text skew angle and rotates the image to straighten it."""
    coords = np.column_stack(np.where(img < 255))

    if coords.size == 0:
        return img

    angle = cv2.minAreaRect(coords)[-1]

    angle = -(90 + angle) if angle < -45 else -angle

    h, w = img.shape

    matrix = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        angle,
        1.0,
    )

    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )