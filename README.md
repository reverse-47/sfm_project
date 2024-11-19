# Structure from Motion Implementation

This repository contains an implementation of Structure from Motion (SfM) pipeline for 3D reconstruction from multiple views. The project is part of the Computer Vision course at Nanyang Technological University.

## Overview

The SfM pipeline consists of several steps:
- Feature detection and matching using SIFT
- Baseline pose estimation
- Point cloud triangulation
- New view incorporation
- 3D point cloud visualization

## Requirements

Create a virtual environment and install the required dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

Required packages include:
- OpenCV
- NumPy
- Matplotlib
- Open3D (for visualization)

## Directory Structure

```
.
├── data/
│   ├── fountain-P11/
│   │   ├── images/
│   │   │   ├── 0001.jpg
│   │   │   └── ...
│   │   └── gt_dense_cameras/
│   │       ├── 0000.jpg.camera
│   │       └── ...
│   ├── Herz-Jesus-P8/
│   └── Castle-P19/
├── feature.py
├── main.py
├── utils.py
├── visualize.py
└── requirements.txt
```

## Dataset Preparation

1. Create a `data` directory in the project root
2. Place your image datasets following the structure shown above
3. Supported datasets:
   - fountain-P11
   - Herz-Jesus-P8
   - Castle-P19
   - And more...

## Running the Pipeline

1. First, run feature detection and matching:
```bash
python feature.py --data_dir ./data/ --dataset fountain-P11
```

2. Then, run the main SfM reconstruction:
```bash
python main.py --data_dir ./data/ --dataset fountain-P11
```

### Important Parameters

- `--data_dir`: Root directory containing input data (default: ./data/)
- `--dataset`: Name of dataset (default: Herz-Jesus-P25)
- `--features`: Feature algorithm to use [SIFT|SURF] (default: SURF)
- `--matcher`: Matching algorithm [BFMatcher|FlannBasedMatcher] (default: BFMatcher)
- `--calibration_mat`: Camera calibration type [benchmark|lg_g3] (default: benchmark)
- `--plot_error`: Whether to plot reprojection errors (default: False)

## Output

The reconstruction results will be saved in the following structure:
```
results/
└── dataset_name/
    ├── point-clouds/
    │   └── final_point_cloud.ply
    └── errors/
        └── reprojection_error_plots.png
```

- Point cloud is saved in PLY format and can be viewed using software like MeshLab or CloudCompare
- If `--plot_error` is enabled, reprojection error visualizations will be saved for each camera

## Visualization

The pipeline includes real-time visualization using Open3D. When running `main.py`, a visualization window will appear showing the reconstructed point cloud. You can:
- Rotate the view using the mouse
- Zoom in/out using the scroll wheel
- Pan using Shift + left mouse button

## Results

[Insert your reconstruction result image here]

## Acknowledgments

This project is part of the Computer Vision course at Nanyang Technological University. The implementation is based on classical SfM techniques and uses the following open-source libraries:
- OpenCV for feature detection and matching
- Open3D for point cloud visualization
- NumPy for numerical computations