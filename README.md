# Motion Detection (Problem Set 4)

## Description

This repository contains code supporting Problem Set 4 (Motion Detection / Optical Flow). The goal is to compute a dense flow field that represents motion between frames as a vector field <u(x,y), v(x,y)>.

## What is in this repository

- Source code and scripts to compute and visualize optical flow for image sequences and videos.
- Utility scripts for preprocessing frames and saving results.
- Example notebooks / demonstration scripts (if present) that show how to run the algorithms and visualize output.

Note: Specific filenames and implementation details may vary by submission; inspect the repository files to see the exact scripts included.

## Dependencies

The code was developed and tested with the following (approximate) environment:

- Python 3.8+
- NumPy
- OpenCV (cv2)
- Matplotlib

Install dependencies with pip, for example:

```
pip install numpy opencv-python matplotlib
```

## Usage

Typical usage steps:

1. Place your input image sequence or video files in a local folder (for example `data/`).
2. Run the main script or notebook that computes optical flow. For example:

```
python compute_optical_flow.py --input data/video.mp4 --output results/flow.npy
```

3. Use provided visualization scripts/notebooks to inspect the computed flow fields.

Adjust commands to match the exact filenames in this repository.

## Input data (IMPORTANT)

Input data (image sequences or videos used by the assignment) is NOT included in this repository. The data files referenced by the assignment are stored separately (for example on the course Google Drive or assignment page). To run the code you must provide your own input files and place them in a folder named `data/` (or update script arguments to point to your data location).

## References

See the assignment Google Doc and Piazza post for problem details and input data links.

## License / Attribution

This repository contains student assignment code. Check with the course/instructor for reuse and redistribution permissions.
