import random

import cv2
import torch
import torchvision
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


def perform_detection_and_nms(model, image, det_conf=0.35, nms_threshold=0.25):
    results = model.predict(
        source=image, 
        conf=det_conf, 
        verbose=False, 
        show=False
    )
    detections = results[0]
    boxes = detections.boxes.xyxy.cpu().numpy()
    class_ids = detections.boxes.cls.cpu().numpy().astype(int)
    scores = detections.boxes.conf.cpu().numpy()
    nms_indices = torchvision.ops.nms(
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


def calculate_f1_score(precision, recall):
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def compute_height_bounds(points):
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    return z_min, z_max


def compute_yaw(points):
    xy = points[:, :2]
    centroid_xy = xy.mean(axis=0)
    cov = np.cov((xy - centroid_xy).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    yaw = np.arctan2(principal[1], principal[0])
    return yaw


def rotate_xy(points, yaw):
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw),  np.cos(yaw)]])
    rotated = (R @ points).T
    return rotated


def rotate_points_and_compute_extents(points, yaw):
    rotated = rotate_xy(points, yaw)
    l = rotated[:,0].max() - rotated[:,0].min()
    w = rotated[:,1].max() - rotated[:,1].min()
    return l, w


def compute_local_corners(extent, yaw):
    l, w = extent
    local_corners = np.array([
        [ l/2,  w/2],
        [ l/2, -w/2],
        [-l/2, -w/2],
        [-l/2,  w/2]
    ])
    return rotate_xy(local_corners.T, yaw)


def compute_corners_3d(corners_xy, z_bounds):
    z_min, z_max = z_bounds
    lower_z = np.full((4,1), z_min)
    upper_z = np.full((4,1), z_max)
    corners_3d = np.vstack([
        np.hstack([corners_xy, lower_z]), 
        np.hstack([corners_xy, upper_z])
    ])
    return corners_3d


def fit_bev_oriented_bounding_box(points):
    xy = points[:, :2]
    centroid_xy = xy.mean(axis=0)
    yaw = compute_yaw(points)
    l, w = rotate_points_and_compute_extents((xy - centroid_xy).T, -yaw)
    center_x, center_y = centroid_xy
    local_corners = compute_local_corners((l, w,), yaw)
    corners_xy = local_corners + np.array([center_x, center_y])
    z_min, z_max = compute_height_bounds(points)
    corners_3d = compute_corners_3d(corners_xy, (z_min, z_max,))
    center_z = (z_min + z_max) / 2
    h = z_max - z_min
    return (np.array([center_x, center_y, center_z]),
            np.array([l, w, h]),
            yaw,
            corners_3d)


def cross2(v, w):
    return v[0]*w[1] - v[1]*w[0]


def compute_intersection(S, E, A, B, eps=1e-8):
    S, E, A, B = map(np.array, (S, E, A, B))
    d1 = E - S
    d2 = B - A
    denom = cross2(d1, d2)
    if abs(denom) < eps:
        return None
    t = cross2(A - S, d2) / denom
    return S + t * d1


def polygon_clip(subject_polygon, clip_polygon):
    subject_polygon = [np.array(p, dtype=float) for p in subject_polygon]
    clip_polygon = [np.array(p, dtype=float) for p in clip_polygon]
    output_polygon = subject_polygon
    for i in range(len(clip_polygon)):
        A = clip_polygon[i]
        B = clip_polygon[(i + 1) % len(clip_polygon)]
        input_list = output_polygon
        output_polygon = []
        if len(input_list) == 0:
            break
        S = input_list[-1]
        for E in input_list:
            def inside(P):
                return cross2(B - A, P - A) >= 0
            if inside(E):
                if not inside(S):
                    inter = compute_intersection(S, E, A, B)
                    if inter is not None:
                        output_polygon.append(inter)
                output_polygon.append(E)
            elif inside(S):
                inter = compute_intersection(S, E, A, B)
                if inter is not None:
                    output_polygon.append(inter)
            S = E
    return output_polygon


def polygon_area(polygon):
    polygon = np.array(polygon)
    x = polygon[:,0]
    y = polygon[:,1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_bev_iou(pred_bev, gt_bev):
    inter_poly = polygon_clip(pred_bev, gt_bev)
    if len(inter_poly) < 3:
        inter_area = 0.0
    else:
        inter_area = polygon_area(inter_poly)
    union_area = polygon_area(gt_bev) + polygon_area(pred_bev) - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def find_bev_matched_gt_box(pred_bev, gt_bevs):
    bev_iou_scores = [compute_bev_iou(pred_bev, gt_bev) for gt_bev in gt_bevs]
    max_bev_iou_score = max(bev_iou_scores)
    matched_gt_bev = bev_iou_scores.index(max_bev_iou_score)
    return max_bev_iou_score, matched_gt_bev


def evaluate_bev_metrics(bbox_bev_labels, bbox_bev_preds, iou_threshold=0.5):
    results = Counter()
    matched_gt_bevs = set()
    for pred_bev in bbox_bev_preds:
        max_bev_iou_score, matched_gt_bev = find_bev_matched_gt_box(pred_bev, bbox_bev_labels)
        if max_bev_iou_score >= iou_threshold:
            if matched_gt_bev not in matched_gt_bevs:
                results['TP'] += 1
                matched_gt_bevs.add(matched_gt_bev)
            else:
                results['FP'] += 1
        else:
            results['FP'] += 1
    results['FN'] = len(bbox_bev_labels) - len(matched_gt_bevs)
    return results


def convert_to_bev(bbox):
    corners = np.asarray(bbox.get_box_points())
    bottom_idxs = np.argsort(corners[:, 2])[:4]
    bottom_corners = corners[bottom_idxs][:, [0, 1]]
    centroid = bottom_corners.mean(axis=0)
    angles = np.arctan2(bottom_corners[:,1]-centroid[1],
                        bottom_corners[:,0]-centroid[0])
    ccw_order = np.argsort(angles)
    return bottom_corners[ccw_order]


def compute_approx_3d_iou(pred_box, gt_box):
    pred_bev = convert_to_bev(pred_box)
    gt_bev = convert_to_bev(gt_box)
    inter_poly = polygon_clip(pred_bev, gt_bev)
    if len(inter_poly) < 3:
        inter_area = 0.0
    else:
        inter_area = polygon_area(inter_poly)
    union_area = polygon_area(gt_bev) + polygon_area(pred_bev) - inter_area
    if union_area == 0:
        return 0.0
    bev_iou = inter_area / union_area
    pred_center_z = pred_box.center[2]
    gt_center_z = gt_box.center[2]
    pred_height = pred_box.extent[2]
    gt_height = gt_box.extent[2]
    pred_min_z = pred_center_z - pred_height / 2.0
    pred_max_z = pred_center_z + pred_height / 2.0
    gt_min_z = gt_center_z - gt_height / 2.0
    gt_max_z = gt_center_z + gt_height / 2.0
    inter_height = max(0.0, min(pred_max_z, gt_max_z) - max(pred_min_z, gt_min_z))
    union_height = max(pred_max_z, gt_max_z) - min(pred_min_z, gt_min_z)
    if union_height == 0:
        return 0.0
    height_ratio = inter_height / union_height
    return bev_iou * height_ratio


def find_approx_3d_iou_matched_gt_box(pred_box, bbox_labels):
    approx_3d_iou_scores = [compute_approx_3d_iou(pred_box, gt_box) for gt_box in bbox_labels]
    max_approx_3d_iou_score = max(approx_3d_iou_scores)
    matched_gt_box = approx_3d_iou_scores.index(max_approx_3d_iou_score)
    return max_approx_3d_iou_score, matched_gt_box


def evaluate_approx_3d_iou_with_height(bbox_labels, bbox_preds, iou_threshold=0.5):
    results = Counter()
    matched_gt_boxes = set()
    for pred_box in bbox_preds:
        max_approx_3d_iou_score, matched_gt_box = find_approx_3d_iou_matched_gt_box(pred_box, bbox_labels)
        if max_approx_3d_iou_score >= iou_threshold:
            if matched_gt_box not in matched_gt_boxes:
                results['TP'] += 1
                matched_gt_boxes.add(matched_gt_box)
            else:
                results['FP'] += 1
        else:
            results['FP'] += 1
    results['FN'] = len(bbox_labels) - len(matched_gt_boxes)
    return results
