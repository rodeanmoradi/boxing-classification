import cv2
import tensorflow as tf
import numpy as np

# Turn original image tensor into input format
def preprocess(frame):
    img = cv2.resize(frame, (256, 256))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    return img

cam = cv2.VideoCapture(0)

# Load MoveNet Thunder
path = 'models/movenet/movenet_thunder.tflite'
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()

# Display webcame
while True:
    ret, frame = cam.read()
    
    preprocess(frame)

    cv2.imshow('MacBook Camera', frame)

    if cv2.waitKey(1) == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()