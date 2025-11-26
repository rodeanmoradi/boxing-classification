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
lastFrame = 0
while True:
    ret, frame = cam.read()
    
    if not ret:
        print("Failed to read")
        break

    img = preProcess(frame)
    output = runInference(inputDetails, outputDetails)

    # Draw keypoints 
    confidenceThreshold = 0.3
    lastFrame = curFrame
    curFrame = (curFrame + 1) % N
    for i in range(17):
        point = output[0, 0, i, :]
        yPoint = point[0] * camHeight
        xPoint = point[1] * camWidth
        confidence = point[2]
        
        if confidence > confidenceThreshold:
            ringBuffer[curFrame, i, 0] = xPoint
            ringBuffer[curFrame, i, 1] = yPoint
        else:
            ringBuffer[curFrame, i, 0] = ringBuffer[lastFrame, i, 0]
            ringBuffer[curFrame, i, 1] = ringBuffer[lastFrame, i, 1]
    smoothed = np.mean(ringBuffer, axis=0)

    for i in range(17):
        cv2.circle(frame, (int(smoothed[i][0]), int(smoothed[i][1])), 7, (255, 0, 0), -1)

    cv2.imshow('MacBook Camera', frame)

    if cv2.waitKey(1) == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()