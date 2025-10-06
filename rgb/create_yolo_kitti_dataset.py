import os
import random
import shutil
import cv2

CLASS_MAP = {
    "Car": 0,
    "Pedestrian": 1,
    "Cyclist": 2,
}

src_labels = "/home/tom/kitti/training/label_2"
src_images = "/home/tom/kitti/training/image_2"
dst_labels_train = "/home/tom/datasets/kitti/labels/train"
dst_images_train = "/home/tom/datasets/kitti/images/train"
dst_labels_val = "/home/tom/datasets/kitti/labels/val"
dst_images_val = "/home/tom/datasets/kitti/images/val"

# create directories
for d in [dst_labels_train, dst_images_train, dst_labels_val, dst_images_val]:
    os.makedirs(d, exist_ok=True)

# shuffle and split
all_labels = os.listdir(src_labels)
random.shuffle(all_labels)
split_idx = int(0.8 * len(all_labels))
train_labels = all_labels[:split_idx]
val_labels = all_labels[split_idx:]

def process_and_copy(label_list, dst_labels, dst_images):
    for label_file in label_list:
        img_file = label_file.replace(".txt", ".png")
        with open(os.path.join(src_labels, label_file)) as f:
            lines = f.readlines()

        yolo_lines = []
        img_path = os.path.join(src_images, img_file)
        h, w, _ = cv2.imread(img_path).shape
        for line in lines:
            parts = line.strip().split()
            cls = parts[0]
            if cls not in CLASS_MAP:
                continue
            cls_id = CLASS_MAP[cls]
            x1, y1, x2, y2 = map(float, parts[4:8])
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            width = (x2 - x1) / w
            height = (y2 - y1) / h
            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        with open(os.path.join(dst_labels, label_file), "w") as out:
            out.writelines(yolo_lines)

        shutil.copy(img_path, os.path.join(dst_images, img_file))

# process train and val
process_and_copy(train_labels, dst_labels_train, dst_images_train)
process_and_copy(val_labels, dst_labels_val, dst_images_val)
