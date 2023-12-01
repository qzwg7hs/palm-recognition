# palm-recognition

This project is an implementation of a palm recognition system using Python. It leverages image processing techniques and mathematical algorithms for distance measure to identify matches among hand palm images.

## Features

- Palm image acquisition and preprocessing
- Feature extraction (histograms of texture and gradient feature images, blockwise area, GLCM numerical features)
- Matching (Euclidean distance, Cosine distance, Chi-squared distance)

## Requirements

To run this project, you need the following dependencies:
- Python (version 3.7 or higher)
- OpenCV
- NumPy
- Pandas
- Scikit-image

You can install these dependencies using pip:

```shell
pip install opencv-python numpy pandas scikit-image
