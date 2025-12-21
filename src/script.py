import cv2
import tensorflow as tf
import numpy as np
import time

# TODO: Make config file
PI = 3.14
cam = cv2.VideoCapture(1) # Pass 0 or 1
frame_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
prev_time = time.time()

class PoseEstimator:

    def __init__(self, path):
        self.interpreter = tf.lite.Interpreter(model_path=path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame, (192, 192))
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)

        return img
    
    def inference(self, img):
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output
    
    def detect(self, frame):
        img = self.preprocess(frame)
        output = self.inference(img)

        return output

class SmoothingFilter:

    def __init__(self):
        self.confidence_threshold = 0.3
        self.b = 5.0
        self.f_c_min = 1.0
        self.f_c_d = 0.5
        self.prev_point = np.zeros((1, 17, 2))
        self.prev_speed = np.zeros((1, 17, 2))
        self.smoothed = np.zeros((1, 17, 2))

    # TODO: Tune and optimize
    def filter(self, output, frame_height, frame_width, sampling_period):
        
        if sampling_period <= 0.00001: 
            sampling_period = 0.016
        
        for i in range(17):
            point = output[0, 0, i, :]
            y_point = point[0] * frame_height
            x_point = point[1] * frame_width
            confidence = point[2]
        
            if confidence > self.confidence_threshold:

                # Smooth speed
                x_dot_0 = (x_point - self.prev_point[0, i, 0]) / sampling_period
                x_dot_1 = (y_point - self.prev_point[0, i, 1]) / sampling_period
                a_d = 1 / (1 + (1 / (2 * PI * self.f_c_d * sampling_period)))
                x_dot_hat_0 = x_dot_0 * a_d + self.prev_speed[0, i, 0] * (1 - a_d)
                x_dot_hat_1 = x_dot_1 * a_d + self.prev_speed[0, i, 1] * (1 - a_d)

                # Smooth position 
                f_c_0 = self.f_c_min + self.b * abs(x_dot_hat_0)
                a_0 = 1 / (1 + (1 / (2 * PI * f_c_0 * sampling_period)))
                x_hat_0 = x_point * a_0 + self.smoothed[0, i, 0] * (1 - a_0)
                f_c_1 = self.f_c_min + self.b * abs(x_dot_hat_1)
                a_1 = 1 / (1 + (1 / (2 * PI * f_c_1 * sampling_period)))
                x_hat_1 = y_point * a_1 + self.smoothed[0, i, 1] * (1 - a_1)

                self.smoothed[0, i, 0] = x_hat_0
                self.smoothed[0, i, 1] = x_hat_1

                self.prev_speed[0, i, 0] = x_dot_hat_0
                self.prev_speed[0, i, 1] = x_dot_hat_1
                
                self.prev_point[0, i, 0] = x_point
                self.prev_point[0, i, 1] = y_point 

        return self.smoothed

class Visualiser:
    
    def __init__(self):
        self.connections = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (5, 6),
            (5, 7),
            (5, 11),
            (6, 8),
            (6, 12),
            (7, 9),
            (8, 10),
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16)
        ]
        
    def draw_keypoints(self, frame, smoothed):
        for (start, end) in self.connections:
            cv2.line(frame, (int(smoothed[0, start, 0]), int(smoothed[0, start, 1])), 
                     (int(smoothed[0, end, 0]), int(smoothed[0, end, 1])), (255, 0, 0), 3)
        for i in range(17):
            cv2.circle(frame, (int(smoothed[0, i, 0]), int(smoothed[0, i, 1])), 8, (255, 0, 0), -1)

movenet_lightning = PoseEstimator('models/movenet/movenet_lightning.tflite')
moving_average = SmoothingFilter()
visualiser = Visualiser()

# TODO: Make main function
while True:
    current_time = time.time()
    frame_time = current_time - prev_time
    prev_time = current_time
    
    ret, frame = cam.read()
    
    if not ret:
        print("Failed to read")
        break

    smoothed = moving_average.filter(movenet_lightning.detect(frame), frame_height, frame_width, frame_time)

    visualiser.draw_keypoints(frame, smoothed)

    cv2.imshow('MacBook Camera', frame)

    if cv2.waitKey(1) == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()