# Real-Time Boxing Punch Classification

A computer vision system that detects punches in real-time using pose estimation and deep learning. The system captures video from a webcam, extracts skeletal keypoints using MoveNet, smoothes the data with a One Euro filter, and classifies action sequences using a custom PyTorch LSTM. At the moment, the model is capable of classifying jabs and uppercuts.

# Demo



https://github.com/user-attachments/assets/5d2aeee1-7c83-49ae-8768-1b60b96f2385



# System Architecture

1.  **Pose Estimation**: MoveNet Lightning (TFLite) extracts 17 keypoints (x, y) from the video feed.
2.  **Signal Processing**: A One Euro Filter removes high-frequency jitter from the raw keypoints to prevent noise from being misinterpreted as movement.
3.  **Temporal Buffering**: A circular buffer maintains a sliding window of the last 30 frames.
4.  **Classification**: A custom PyTorch LSTM analyzes the temporal sequence (1, 30, 34) and outputs logits. Using the softmax function, these logits are converted to probabilities that allow a classification to be made.

# Stack

* **Language**: Python 3.9+
* **Deep Learning**: PyTorch (LSTM training & inference)
* **Pose Estimation**: TensorFlow Lite (MoveNet)
* **Computer Vision**: OpenCV (Video capture & visualization)
* **Data Handling**: NumPy (Buffer management & smoothing filter)
