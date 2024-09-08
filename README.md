# Face Mask Detection System

This repository contains a set of Python scripts for detecting faces and predicting whether they are wearing a mask or not using deep learning models. The system is built using TensorFlow, Keras, and OpenCV. 

## Prerequisites:

Before running the scripts, ensure you have the following installed:

- Python 3.x
- TensorFlow
- Keras
- OpenCV
- NumPy
- imutils
- Matplotlib
- scikit-learn

You can install the required packages using pip:

```bash
pip install tensorflow keras opencv-python-headless numpy imutils matplotlib scikit-learn
```

## Code Overview:

### 1. Real-Time Video Mask Detection (`detect_mask_video.py`)

This script performs real-time face mask detection using a webcam. It captures video frames, detects faces, and predicts whether each detected face is wearing a mask.

#### Usage

1. Ensure the `face_detector` directory contains `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` files for face detection.
2. Place the pre-trained mask detection model file `mask_detector.model` in the same directory as this script.
3. Run the script:

    ```bash
    python video_mask_detection.py
    ```

4. The video stream will open, and faces will be detected and labeled as either "Face-Detected with mask" or "Face-Detected with No Mask".

### 2. Image Mask Detection (`detect_pic.py`):

This script performs face mask detection on a single input image. It loads the image, detects faces, predicts mask usage, and saves or displays the output image with labels.

#### Usage

1. Ensure the `face_detector` directory contains `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` files for face detection.
2. Place the pre-trained mask detection model file `mask_detector.model` in the same directory as this script.
3. Replace `"12.jpg"` with the path to your input image.
4. Run the script:

    ```bash
    python image_mask_detection.py
    ```

5. The output image will be saved as `output.jpg` and displayed with labels.

### 3. Model Training (`train_mask_detector.py`)

This script trains a mask detection model using the MobileNetV2 architecture. It prepares the dataset, trains the model, and evaluates its performance.

#### Usage

1. Place your dataset in the `DIRECTORY` path specified in the script. The dataset should be organized into two folders: `with_mask` and `without_mask`.
2. Run the script to train the model and save it as `mask_detector.model`:

    ```bash
    python train_model.py
    ```

3. The script will plot the training loss and accuracy, and save the plot as `plot.png`.

## Files

- `video_mask_detection.py`: Real-time mask detection using a webcam.
- `image_mask_detection.py`: Mask detection on a single image.
- `train_model.py`: Model training and evaluation script.



## Acknowledgements

- TensorFlow and Keras for deep learning functionalities.
- OpenCV for computer vision operations.
- MobileNetV2 for transfer learning.

