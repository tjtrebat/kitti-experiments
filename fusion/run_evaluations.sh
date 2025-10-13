#!/bin/bash

NUM_SAMPLES=1000

echo "=== Full 3D IoU, axis-aligned ==="
python evaluate.py -n $NUM_SAMPLES

echo "=== Full 3D IoU, oriented ==="
python evaluate.py --oriented -n $NUM_SAMPLES

echo "=== BEV IoU, axis-aligned ==="
python evaluate.py --bev -n $NUM_SAMPLES

echo "=== BEV IoU, oriented ==="
python evaluate.py --bev --oriented -n $NUM_SAMPLES

