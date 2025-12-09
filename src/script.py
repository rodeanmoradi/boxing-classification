import cv2
import tensorflow as tf
import numpy as np

# TODO: Make config file
# Pass 0 or 1 if it fails to read
cam = cv2.VideoCapture(0)
frame_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)

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
        self.a = 0.5
        self.smoothed = np.zeros((1, 17, 2))

    # TODO: Switch to One Euro filter
    def filter(self, output, frame_height, frame_width):
        for i in range(17):
            point = output[0, 0, i, :]
            yPoint = point[0] * frame_height
            xPoint = point[1] * frame_width
            confidence = point[2]
        
            if confidence > self.confidence_threshold:
                self.smoothed[0, i, 0] = (xPoint * self.a) + self.smoothed[0, i, 0] * (1 - self.a)
                self.smoothed[0, i, 1] = (yPoint * self.a) + self.smoothed[0, i, 1] * (1 - self.a)

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

movenet_thunder = PoseEstimator('models/movenet/movenet_lightning.tflite')
moving_average = SmoothingFilter()
visualiser = Visualiser()

while True:
    ret, frame = cam.read()
    
    if not ret:
        print("Failed to read")
        break

    output = movenet_thunder.detect(frame)

    smoothed = moving_average.filter(output, frame_height, frame_width)

    visualiser.draw_keypoints(frame, smoothed)

    cv2.imshow('MacBook Camera', frame)

    if cv2.waitKey(1) == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()