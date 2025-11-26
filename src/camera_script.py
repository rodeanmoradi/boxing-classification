import cv2
import tensorflow as tf
import numpy as np

def preProcess(frame):
    img = cv2.resize(frame, (256, 256))
    img = np.expand_dims(img, axis=0)

    return img

def runInference(inputDetails, outputDetails):
    interpreter.set_tensor(inputDetails[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(outputDetails[0]['index'])

    return output

# Might have to switch to cv2.VideoCapture(0) if fails to read
cam = cv2.VideoCapture(1)
camHeight = 720
camWidth = 1280

# Deploy MoveNet Thunder model, TODO: def deploy() that returns in, out details
path = 'models/movenet/movenet_thunder.tflite'
interpreter = tf.lite.Interpreter(model_path=path)
interpreter.allocate_tensors()
inputDetails = interpreter.get_input_details()
outputDetails = interpreter.get_output_details()

N = 5
ringBuffer = np.zeros((N, 17, 2))
curFrame = 0
while True:
    ret, frame = cam.read()
    
    if not ret:
        print("Failed to read")
        break

    img = preProcess(frame)
    output = runInference(inputDetails, outputDetails)

    #Draw keypoints, TODO: implement moving avg for smoothing
    confidenceThreshold = 0.3
    for i in range(17):
        point = output[0, 0, i, :]
        yPoint = int(point[0] * camHeight)
        xPoint = int(point[1] * camWidth)
        confidence = point[2]
        if confidence > confidenceThreshold:
            cv2.circle(frame, (xPoint, yPoint), 5, (255, 0, 0), -1)
        ringBuffer[curFrame, i, 0] = xPoint
        ringBuffer[curFrame, i, 1] = yPoint
    lastFrame = curFrame
    curFrame = (lastFrame + 1) % N

    cv2.imshow('MacBook Camera', frame)

    if cv2.waitKey(1) == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()