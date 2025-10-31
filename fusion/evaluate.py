import sys
import time
import random
import argparse

import yaml
import numpy as np
import pandas as pd
import open3d as o3d

import utils
import calibration_utils

from pathlib import Path
from collections import Counter

from ultralytics import YOLO


parser = argparse.ArgumentParser(description="Evaluate object proposals")
parser.add_argument(
    "--bev",
    action="store_true",
    help="Use BEV IoU evaluation instead of full 3D IoU"
)
parser.add_argument(
    "--oriented",
    action="store_true",
    help="Use oriented bounding boxes instead of axis-aligned ones"
)
parser.add_argument(
    "-n", "--num-samples",
    type=int,
    default=None,
    help="Limit the number of samples used for evaluation (default: use all)"
)
parser.add_argument(
    "--yolo-weights",
    type=str,
    default=None,
    help="Path to YOLO pretrained weights to generate 2D bounding box predictions"
)
args = parser.parse_args()


np.random.seed(42)
random.seed(42)
o3d.utility.random.seed(42)

MAX_POINTS = 300
MIN_POINTS = 25
OBJECT_EXTENT_STATS = {
    'Car': {
        'height': {'mean': 1.526083, 'std': 0.136697},
        'width': {'mean': 1.628590, 'std': 0.102164},
        'length': {'mean': 3.883954, 'std': 0.425924}
    },
    'Pedestrian': {
        'height': {'mean': 1.760706, 'std': 0.113263},
        'width': {'mean': 0.660189, 'std': 0.142667},
        'length': {'mean': 0.842284, 'std': 0.234926}
    },
    'Cyclist': {
        'height': {'mean': 1.737203, 'std': 0.094825},
        'width': {'mean': 0.596773, 'std': 0.124212},
        'length': {'mean': 1.763546, 'std': 0.176663}
    }
}


def get_point_cloud(points_velo):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_velo[:, :3])
    return cloud


def downsample(cloud):
    cloud = cloud.voxel_down_sample(voxel_size=0.2)
    return cloud


def segment(cloud):
    _, inliers = cloud.segment_plane(
        distance_threshold=0.3, 
        ransac_n=3, 
        num_iterations=150
    )
    outlier_cloud = cloud.select_by_index(inliers, invert=True)
    return outlier_cloud


def get_downsampled_point_cloud(points_velo):
    cloud = get_point_cloud(points_velo)
    cloud = downsample(cloud)
    cloud = segment(cloud)
    return cloud


def depth_filter(cloud):
    cloud_points = np.array(cloud.points)
    mask_in_front = cloud_points[:, 0] > 0
    cloud_points_in_cam_front = cloud_points[mask_in_front]
    return cloud_points_in_cam_front


def get_points_uv(cloud_points_in_cam_front, calib):
    points_uv = calibration_utils.convert_to_img_coords(
        cloud_points_in_cam_front, 
        calib
    )
    return points_uv[:, :2]


def get_labels(label_file_path):
    labels = utils.parse_label_file(label_file_path)
    labels = [label for label in labels 
              if label['type'] in ['Car', 'Pedestrian', 'Cyclist']]
    return labels


def load_yolo_model(weights_path: str, yaml_path: str):
    try:
        model = YOLO(weights_path)
    except Exception as e:
        sys.exit(f"Failed to load YOLO model from '{weights_path}': {e}")
    try:
        with open(yaml_path, 'r') as stream:
            class_names = yaml.safe_load(stream)['names']
    except FileNotFoundError:
        sys.exit(f"YAML file for class names not found: {yaml_path}")
    except yaml.YAMLError as exc:
        sys.exit(f"Error parsing YAML file '{yaml_path}': {exc}")
    print(f"Loaded YOLO model from '{weights_path}' with classes: {class_names}")
    return model, class_names


def perform_detection_and_nms(model, image):
    det_boxes, det_class_ids, det_scores = utils.perform_detection_and_nms(
        model, 
        image, 
        det_conf=0.10, 
        nms_threshold=0.50
    )
    if det_boxes.ndim == 1:
        det_boxes = np.reshape(det_boxes, (1, 4))
        det_class_ids = [det_class_ids]
        det_scores = [det_scores]
    return det_boxes, det_class_ids, det_scores


def get_mask_frustums(points_uv, bbox_2d_labels):
    mask_frustums = []
    u, v = points_uv[:, 0], points_uv[:, 1]
    valid_mask = np.isfinite(u) & np.isfinite(v)
    for xmin, ymin, xmax, ymax in bbox_2d_labels:
        mask_frustum = (u >= xmin) & (u <= xmax) & (v >= ymin) & (v <= ymax)
        mask_frustum = mask_frustum & valid_mask
        mask_frustums.append(mask_frustum)
    return mask_frustums


def extract_frustum_points(mask_frustums, cloud_points_in_cam_front):
    frustum_points = []
    for mask_frustum in mask_frustums:
        points = cloud_points_in_cam_front[mask_frustum]
        frustum_points.append(points)
    return frustum_points


def get_frustum(points):
    frustum = o3d.geometry.PointCloud()
    frustum.points = o3d.utility.Vector3dVector(points[:, :3])
    return frustum


def get_cluster_labels(frustum):
    if len(frustum.points) == 0:
        return np.array([])
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning) as cm:
        cluster_labels = np.array(
            frustum.cluster_dbscan(
                eps=0.45, 
                min_points=10, 
                print_progress=False
            )
        )
    return cluster_labels


def group_cluster_labels(cluster_labels):
    valid_mask = cluster_labels != -1
    return (
        pd.Series(np.arange(len(cluster_labels)))[valid_mask]
        .groupby(cluster_labels[valid_mask], sort=False)
        .apply(list)
        .tolist()
    )


def get_frustum_with_clusters(points):
    frustum = get_frustum(points)
    cluster_labels = get_cluster_labels(frustum)
    return frustum, group_cluster_labels(cluster_labels)    


def get_frustums_with_clusters(frustum_points):
    frustums_with_clusters = []
    for i, points in enumerate(frustum_points):
        frustum, clusters = get_frustum_with_clusters(points)
        frustums_with_clusters.append((frustum, clusters,))
    return frustums_with_clusters


def get_axis_aligned_bounding_box(cluster):
    axis_aligned_box = cluster.get_axis_aligned_bounding_box()
    return axis_aligned_box


def get_axis_aligned_bounding_boxes(proposals):
    axis_aligned_boxes = []
    for cluster in proposals:
        axis_aligned_box = get_axis_aligned_bounding_box(cluster)
        axis_aligned_boxes.append(axis_aligned_box)
    return axis_aligned_boxes


def get_axis_aligned_bounding_box_proposals(frustum, clusters):
    proposals = []
    for cluster_idx in clusters:
        cluster = frustum.select_by_index(cluster_idx)
        if MIN_POINTS < len(cluster.points) < MAX_POINTS:
            proposals.append(cluster)
    return proposals    


def get_axis_aligned_bounding_box_predictions(frustums_with_clusters):
    axis_aligned_box_predictions = []
    for frustum, clusters in frustums_with_clusters:
        proposals = get_axis_aligned_bounding_box_proposals(frustum, clusters)
        axis_aligned_boxes = get_axis_aligned_bounding_boxes(proposals)
        axis_aligned_box_predictions.extend(axis_aligned_boxes)
    return axis_aligned_box_predictions 


def get_cam_to_velo_transform(calib):
    Tr_velo_to_cam = calib['Tr_velo_to_cam']
    Tr_velo_to_cam = np.vstack((Tr_velo_to_cam, [0, 0, 0, 1]))
    Tr_cam_to_velo = calibration_utils.inverse_rigid_transform(Tr_velo_to_cam)
    return Tr_cam_to_velo


def get_axis_aligned_box_labels(labels, Tr_cam_to_velo):
    axis_aligned_box_labels = []
    for label in labels:
        corners_3d_velo = calibration_utils.compute_box_3d(label, Tr_cam_to_velo)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(corners_3d_velo)
        axis_aligned_box_label = get_axis_aligned_bounding_box(cloud)
        axis_aligned_box_labels.append(axis_aligned_box_label)
    return axis_aligned_box_labels


def get_oriented_bounding_box(proposal):
    center, extent, yaw, corners3d = proposal
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw),  np.cos(yaw), 0],
                  [0,            0,           1]])
    oriented_bbox = o3d.geometry.OrientedBoundingBox(center, R, extent)
    return oriented_bbox


def get_oriented_bounding_boxes(proposals):
    oriented_bboxes = []
    for proposal in proposals:
        oriented_bbox = get_oriented_bounding_box(proposal)
        oriented_bboxes.append(oriented_bbox)
    return oriented_bboxes


def score_proposal(extent, object_type, z_threshold=2.0):
    dim_names = ['length', 'width', 'height']
    scores = []
    for value, dim_name in zip(extent, dim_names):
        mean = OBJECT_EXTENT_STATS[object_type][dim_name]['mean']
        std = OBJECT_EXTENT_STATS[object_type][dim_name]['std']
        z = np.abs(value - mean) / std
        dim_score = max(0, 1 - z / z_threshold)
        scores.append(dim_score)
    return np.mean(scores)


def get_oriented_bounding_box_proposals(frustum_with_clusters, object_type):
    proposals = []
    frustum, clusters = frustum_with_clusters
    for cluster_idx in clusters:
        cluster = frustum.select_by_index(cluster_idx)
        points = np.asarray(cluster.points)
        center, extent, yaw, corners3d = utils.fit_bev_oriented_bounding_box(points)
        if score_proposal(extent, object_type) > 0:
            proposals.append((center, extent, yaw, corners3d,))
    return proposals


def get_oriented_bounding_box_predictions(frustums_with_clusters, object_types):
    oriented_bbox_predictions = []
    for i, frustum_with_clusters in enumerate(frustums_with_clusters):
        object_type = object_types[i]
        proposals = get_oriented_bounding_box_proposals(frustum_with_clusters, object_type)
        oriented_bboxes = get_oriented_bounding_boxes(proposals)
        oriented_bbox_predictions.extend(oriented_bboxes)
    return oriented_bbox_predictions


def get_oriented_bounding_box_labels(labels, Tr_cam_to_velo):
    oriented_bbox_labels = []
    for label in labels:
        corners_3d_velo = calibration_utils.compute_box_3d(label, Tr_cam_to_velo)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(corners_3d_velo)
        oriented_bbox = cloud.get_oriented_bounding_box()
        oriented_bbox_labels.append(oriented_bbox)
    return oriented_bbox_labels


def evaluate_box_iou(bbox_labels, bbox_preds):
    results = utils.evaluate_metrics(
        bbox_labels, 
        bbox_preds, 
        iou_threshold=0.2
    )
    return results


def evaluate_bev_iou(bbox_bev_labels, bbox_bev_preds):
    results = utils.evaluate_bev_metrics(
        bbox_bev_labels, 
        bbox_bev_preds, 
        iou_threshold=0.2
    )
    return results


def evaluate_approx_3d_iou_with_height(bbox_labels, bbox_preds):
    results = utils.evaluate_approx_3d_iou_with_height(
        bbox_labels, 
        bbox_preds, 
        iou_threshold=0.2
    )
    return results


base = Path.home() / "kitti"
train_dir = base / "training"

train_labels_dir = train_dir / "label_2"
kitti_train_labels = sorted(train_labels_dir.glob("*.txt"))

train_img_dir = train_dir / "image_2"
kitti_images_train = sorted(train_img_dir.glob("*.png"))

velo_dir = train_dir / "velodyne"
point_cloud_train_files = sorted(velo_dir.glob("*.bin"))

calib_dir = train_dir / "calib"
calib_train_files = sorted(calib_dir.glob("*.txt"))

dataset_size = len(kitti_train_labels)
num_samples = args.num_samples or dataset_size
num_samples = min(num_samples, dataset_size)
random_indices = random.sample(range(dataset_size), num_samples)

yaml_path = Path.home() / "datasets/kitti/data.yaml"
if args.yolo_weights:
    fine_tuned_model, tuned_model_class_names = load_yolo_model(
        args.yolo_weights, 
        yaml_path
    )

results = Counter()
start_time = time.time()

for analysis_file_index in random_indices:
    bin_path = point_cloud_train_files[analysis_file_index]
    points_velo = utils.read_velodyne_bin(bin_path)
    cloud = get_downsampled_point_cloud(points_velo)
    cloud_points_in_cam_front = depth_filter(cloud)
    calib = utils.parse_calib_file(calib_train_files[analysis_file_index])
    points_uv = get_points_uv(cloud_points_in_cam_front, calib)
    labels = get_labels(kitti_train_labels[analysis_file_index])
    if args.yolo_weights:
        det_boxes, det_class_ids, _ = perform_detection_and_nms(
            fine_tuned_model,
            kitti_images_train[analysis_file_index]
        )
        bbox_2d_labels = [box.tolist() for box in det_boxes]
        object_types = [tuned_model_class_names[class_id] for class_id in det_class_ids]
    else:
        bbox_2d_labels = [label['bbox_2d'] for label in labels] 
        object_types = [label['type'] for label in labels]
    mask_frustums = get_mask_frustums(points_uv, bbox_2d_labels)
    frustum_points = extract_frustum_points(mask_frustums, cloud_points_in_cam_front)
    frustums_with_clusters = get_frustums_with_clusters(frustum_points)
    Tr_cam_to_velo = get_cam_to_velo_transform(calib)
    if args.oriented:
        bbox_preds = get_oriented_bounding_box_predictions(frustums_with_clusters, object_types)
        bbox_labels = get_oriented_bounding_box_labels(labels, Tr_cam_to_velo)
    else:
        bbox_preds = get_axis_aligned_bounding_box_predictions(frustums_with_clusters)
        bbox_labels = get_axis_aligned_box_labels(labels, Tr_cam_to_velo)
    if args.bev:
        bbox_bev_labels = [utils.convert_to_bev(label) for label in bbox_labels]
        bbox_bev_preds = [utils.convert_to_bev(pred) for pred in bbox_preds]        
        cloud_results = evaluate_bev_iou(bbox_bev_labels, bbox_bev_preds)
    else:
        if args.oriented:
            cloud_results = evaluate_approx_3d_iou_with_height(bbox_labels, bbox_preds)
        else:
            cloud_results = evaluate_box_iou(bbox_labels, bbox_preds)
    results.update(cloud_results)


elapsed = time.time() - start_time
print(f"Total evaluation time: {elapsed:.2f} seconds")

precision, recall = utils.calculate_precision_recall(results)
f1_score = utils.calculate_f1_score(precision, recall)
print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}")
print(f"Raw Results: {dict(results)}")
