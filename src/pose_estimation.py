import tensorflow as tf
import cv2
import numpy as np

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