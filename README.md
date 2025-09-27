# KITTI Point Cloud Visualization Notebook

This notebook demonstrates how to visualize raw KITTI Velodyne point clouds using Open3D in a Jupyter environment.

---

## Requirements

- Python ≥ 3.8  
- Open3D ≥ 0.17  
- NumPy  
- Matplotlib (optional, for fallback visualization)

Install dependencies:

```bash
pip install open3d numpy matplotlib
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

