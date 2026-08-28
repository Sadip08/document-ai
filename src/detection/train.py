from ultralytics import YOLO
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

def train_layout_detector(data_yml_path, epochs=50, img_size=1280):
    """Fine-tune YOLOv8 on the document layout dataset."""

    model = YOLO("yolov8n.pt")

    models_dir = os.path.join(ROOT_DIR, "models")

    print("Starting training...")
    print(f"Dataset config: {data_yml_path}")

    results = model.train(
        data=data_yml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=8,
        workers=4,
        device=0,
        project=models_dir,
        name="layout_detection_exp",
        save=True,
        random_seed=42,
        exist_ok=True,

        # Document-specific augmentation
        hsv_v=0.4,
        degrees=5.0,
        scale=0.3,
        fliplr=0.0,
        flipud=0.0,
    )

    return results


if __name__ == "__main__":

    yaml_path = os.path.join(
        ROOT_DIR,
        "data",
        "splits",
        "document_layout",
        "data.yaml",
    )

    if not os.path.exists(yaml_path):
        print(f"Dataset config not found:\n{yaml_path}")
    else:
        train_layout_detector(yaml_path, epochs=50)