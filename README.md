# KITTI Visualization and Detection Notebook

This notebook demonstrates how to visualize raw KITTI Velodyne point clouds using Open3D in a Jupyter environment, and how to visualize detection results from a fine-tuned YOLOv8 model using SAHI for sliced inference.

---

## Requirements

- Python ≥ 3.8  
- Packages listed in `requirements.txt`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Notebook

1. Set environment variable for X11 (Wayland workaround)

If you are on Linux with Wayland, Open3D’s native rendering may fail. To fix this, set the following environment variable before launching Jupyter:

```bash
export XDG_SESSION_TYPE=x11
jupyter notebook
```

This workaround is described in the Open3D GitHub Issue: [#6872](https://github.com/isl-org/Open3D/issues/6872).

2. Open the Notebook

- Navigate to the notebook in Jupyter.
- Make sure the paths to the KITTI .bin files and images are correct.
- Run the cells. The notebook will load and display the point clouds.

