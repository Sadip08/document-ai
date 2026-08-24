import os
import random
import shutil

test_images_dir = "E:\\document-ai\\data\\splits\\test\\images"
test_labels_dir = "E:\\document-ai\\data\\splits\\test\\labels"
valid_images_dir = "E:\\document-ai\\data\\splits\\valid\\images"
valid_labels_dir = "E:\\document-ai\\data\\splits\\valid\\labels"

os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

images = [f for f in os.listdir(valid_images_dir) if f.endswith('.jpg') or f.endswith('.png')]
random.shuffle(images)

val_split = 0.5
num_val = int(len(images) *  val_split)

print(f"Total test images found: {len(images)}")
print(f"Moving {num_val} images to validation set.")

for i, img_file in enumerate(images[:num_val]):
    src_img_path = os.path.join(valid_images_dir, img_file)
    dst_img_path = os.path.join(test_images_dir, img_file)
    shutil.move(src_img_path, dst_img_path)

    label_file = os.path.splitext(img_file)[0] + '.txt'
    src_label_path = os.path.join(valid_labels_dir, label_file)
    dst_label_path = os.path.join(test_labels_dir, label_file)
    if os.path.exists(src_label_path):
        shutil.move(src_label_path, dst_label_path)
    else:
        print(f"Warning: Label file {label_file} not found for image {img_file}.")