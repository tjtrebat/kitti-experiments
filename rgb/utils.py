import torch
import numpy as np


def perform_detection_and_nms(model, image, det_conf=0.35, nms_threshold=0.25):
    results = model.predict(source=image, conf=det_conf)
    detections = results[0]
    boxes = detections.boxes.xyxy.cpu().numpy()
    class_ids = detections.boxes.cls.cpu().numpy().astype(int)
    scores = detections.boxes.conf.cpu().numpy()
    nms_indices = torch.ops.torchvision.nms(
        torch.tensor(boxes),
        torch.tensor(scores),
        nms_threshold
    )
    filtered_boxes = np.array(boxes[nms_indices])
    filtered_class_ids = class_ids[nms_indices]
    filtered_scores = scores[nms_indices]
    return filtered_boxes, filtered_class_ids, filtered_scores


def parse_label_file(label_file_path):
    parsed_labels = []
    with open(label_file_path, "r") as file:
        for line in file:
            line_elements = line.strip().split()
            if len(line_elements) != 15:
                raise ValueError(f"Line does not match expected format: {line.strip()}")
            label = {
                'type': line_elements[0],
                'truncated': float(line_elements[1]),
                'occluded': int(line_elements[2]),
                'alpha': float(line_elements[3]),
                'bbox_2d_min': tuple(float(bbox_2d_dim) for bbox_2d_dim in line_elements[4:6]),
                'bbox_2d_max': tuple(float(bbox_2d_dim) for bbox_2d_dim in line_elements[6:8]),
                'dimensions': tuple(float(dim) for dim in line_elements[8:11]),
                'centroid': tuple(float(loc) for loc in line_elements[11:14]),
                'rotation_y': float(line_elements[14])
            }
            parsed_labels.append(label)
    return parsed_labels
