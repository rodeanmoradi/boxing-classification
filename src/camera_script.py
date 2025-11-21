import cv2
import tensorflow as tf
import numpy as np

def preProcess(frame):
    img = cv2.resize(frame, (256, 256))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    return img

def runInference(inputDetails, outputDetails):
    interpreter.set_tensor(inputDetails[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(outputDetails[0]['index'])

    return output

cam = cv2.VideoCapture(0)

# Deploy MoveNet Thunder model
path = 'models/movenet/movenet_thunder.tflite'
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()
inputDetails = interpreter.get_input_details()
outputDetails = interpreter.get_output_details()

while True:
    ret, frame = cam.read()
    
    img = preProcess(frame)
    output = runInference(inputDetails, outputDetails)

    #TODO: Use output to draw keypoints


    cv2.imshow('MacBook Camera', frame)

    if cv2.waitKey(1) == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()