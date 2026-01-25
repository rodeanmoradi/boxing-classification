# Real-Time Boxing Punch Classification

A computer vision system that detects boxing punches in real-time using pose estimation and deep learning. The system captures video from a webcam, extracts skeletal keypoints using MoveNet, smoothes the data, and classifies action sequences using a custom PyTorch LSTM. This project is still in progress, and is only capable of classifying jabs.

# System Architecture

1.  **Pose Estimation**: MoveNet Lightning (TFLite) extracts 17 keypoints (x, y) from the video feed.
2.  **Signal Processing**: A One Euro Filter removes high-frequency jitter from the raw keypoints to prevent noise from being misinterpreted as movement.
3.  **Temporal Buffering**: A circular buffer maintains a sliding window of the last 30 frames.
4.  **Classification**: A custom PyTorch LSTM analyzes the temporal sequence (1, 30, 34) and outputs a classification.

# Stack

* **Language**: Python 3.9+
* **Deep Learning**: PyTorch (LSTM Training & Inference)
* **Pose Estimation**: TensorFlow Lite (MoveNet)
* **Computer Vision**: OpenCV (Video capture & Visualization)
* **Data Handling**: NumPy (Buffer management)
