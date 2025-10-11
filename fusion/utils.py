import cv2
import random
import numpy as np

from collections import Counter


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


def draw_detection_output(image, detections, color_rgb=None):
    image_with_detections = image.copy()
    for detection in detections:
        xmin, ymin, xmax, ymax = map(int, detection["bounding_box"])
        label = f"{detection['object_name']} ({detection['confidence']:.2f})"
        color = color_rgb if color_rgb else tuple(random.randint(0, 255) for _ in range(3))
        cv2.rectangle(image_with_detections, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(image_with_detections, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image_with_detections


def read_velodyne_bin(file_path):
    data = np.fromfile(file_path, dtype=np.float32)
    return data.reshape(-1, 4)


def parse_calib_file(calib_file_path):
    calib_keys = ['P0', 'P1', 'P2', 'P3', 'R0','Tr_velo_to_cam', 'Tr_imu_to_velo']
    calibration_matrices = dict()
    with open(calib_file_path, "r") as file:
        calib_lines = file.readlines()
        for i, key in enumerate(calib_keys):
            elems = calib_lines[i].split(' ')
            elems = elems[1:]
            calib_matrix = np.array(elems, dtype=np.float32)
            if key == 'R0':
                calib_matrix_shape = (3, 3)
            else:
                calib_matrix_shape = (3, 4)
            calib_matrix = calib_matrix.reshape(calib_matrix_shape)
            calibration_matrices[key] = calib_matrix
    return calibration_matrices


def compute_iou(box1, box2):
    min1, max1 = np.array(box1.min_bound), np.array(box1.max_bound)
    min2, max2 = np.array(box2.min_bound), np.array(box2.max_bound)
    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)
    intersection_dims = np.maximum(intersection_max - intersection_min, 0)
    intersection_volume = np.prod(intersection_dims)
    volume1 = np.prod(max1 - min1)
    volume2 = np.prod(max2 - min2)
    union_volume = volume1 + volume2 - intersection_volume
    return intersection_volume / union_volume if union_volume > 0 else 0


def find_matched_gt_box(pred_box, gt_boxes):
    iou_scores = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
    max_iou_score = max(iou_scores)
    matched_gt_box = iou_scores.index(max_iou_score)
    return max_iou_score, matched_gt_box


def evaluate_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    results = Counter()
    matched_gt_boxes = set()
    for pred_box in pred_boxes:
        max_iou_score, matched_gt_box = find_matched_gt_box(pred_box, gt_boxes)
        if max_iou_score >= iou_threshold:
            if matched_gt_box not in matched_gt_boxes:
                results['TP'] += 1
                matched_gt_boxes.add(matched_gt_box)
            else:
                results['FP'] += 1
        else:
            results['FP'] += 1
    results['FN'] = len(gt_boxes) - len(matched_gt_boxes)
    return results


def calculate_precision_recall(results):
    TP = results.get('TP', 0)
    FP = results.get('FP', 0)
    FN = results.get('FN', 0)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return precision, recall
