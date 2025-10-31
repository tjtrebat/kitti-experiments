#!/bin/bash

NUM_SAMPLES=2000
YOLO_WEIGHTS="/home/tom/Documents/UNT/csce6260/projects/kitti-experiments/rgb/yolo_results/weights_best.pt"

USE_YOLO=1

YOLO_ARG=""
if [ "$USE_YOLO" -eq 1 ]; then
    YOLO_ARG="--yolo-weights $YOLO_WEIGHTS"
fi

echo "=== Full 3D IoU, axis-aligned ==="
python evaluate.py -n $NUM_SAMPLES $YOLO_ARG

echo "=== Full 3D IoU, oriented ==="
python evaluate.py --oriented -n $NUM_SAMPLES $YOLO_ARG

echo "=== BEV IoU, axis-aligned ==="
python evaluate.py --bev -n $NUM_SAMPLES $YOLO_ARG

echo "=== BEV IoU, oriented ==="
python evaluate.py --bev --oriented -n $NUM_SAMPLES $YOLO_ARG

