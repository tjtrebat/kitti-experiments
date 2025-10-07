import random

import cv2
import torch
import numpy as np

from typing import List, Dict
from collections import Counter


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


def draw_detection_output(image, detections, color_rgb=None):
    image_with_detections = image.copy()
    for detection in detections:
        xmin, ymin, xmax, ymax = map(int, detection["bounding_box"])
        label = f"{detection['object_name']} ({detection['confidence']:.2f})"
        color = color_rgb if color_rgb else tuple(random.randint(0, 255) for _ in range(3))
        cv2.rectangle(image_with_detections, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image_with_detections, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image_with_detections


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def evaluate_detections(pred_detections: List[Dict], gt_detections: List[Dict], iou_threshold=0.5):
    results = Counter()
    for pred_det in pred_detections:
        pred_box = pred_det['bounding_box']
        gt_boxes = [gt_det['bounding_box'] for gt_det in gt_detections]
        ious = [calculate_iou(pred_box, gt_box) for gt_box in gt_boxes]
        best_iou = max(ious)
        matched_gt_idx = ious.index(best_iou)
        if best_iou >= iou_threshold and pred_det['object_name'] == gt_detections[matched_gt_idx]['object_name']:
            results['TP'] += 1
            gt_detections.pop(matched_gt_idx)
        else:
            results['FP'] += 1
    results['FN'] = len(gt_detections)
    return results


def calculate_precision_recall(results):
    TP = results.get('TP', 0)
    FP = results.get('FP', 0)
    FN = results.get('FN', 0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return precision, recall


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
                'bbox_2d': tuple(float(bbox_2d_dim) for bbox_2d_dim in line_elements[4:8]),
                'dimensions': tuple(float(dim) for dim in line_elements[8:11]),
                'centroid': tuple(float(loc) for loc in line_elements[11:14]),
                'rotation_y': float(line_elements[14])
            }
            parsed_labels.append(label)
    return parsed_labels
