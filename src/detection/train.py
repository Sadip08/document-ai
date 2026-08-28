from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

def train_layout_detector(data_yml_path, epochs=50, img_size=1280):
    """
    Fine-tunes a pretrained YOLOv8 model on the document layout dataset.
    """
    model = YOLO('yolov8n.pt')
    print("Starting training")
    print(f"Dataset config : {data_yml_path}")

    # Get the absolute path to the 'models' directory
    # This assumes train.py is inside src/detection/, so ../../ goes back to the root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    models_dir = os.path.join(root_dir, 'models')
    
    results = model.train(
        data = data_yml_path,
        epochs = epochs,
        imgsz = img_size,
        batch = 8,
        device = 0,
        project = models_dir,
        name = "layout_detection_exp",
        save = True,
        exist_ok = True,
        #Augment specific to documents
        hsv_v = 0.4, # adjust brightness
        degrees = 5.0, # adjust rotation
        scale = 0.3, # adjust scale
        fliplr = 0.0, # Turn off horizontal flipping
        flipud = 0.0, # Turn off vertical flipping
    )

    return results

if __name__ == "__main__":
    yaml_path = "E:\\document-ai\\data\\splits\\data.yaml"

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Dataset config file not found at {yaml_path}")
    train_layout_detector(data_yml_path=yaml_path, epochs=50)

