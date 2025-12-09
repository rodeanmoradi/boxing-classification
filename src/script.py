# TODO: Create classes
import cv2
import tensorflow as tf
import numpy as np

# TODO: Make header file
a = 0.5
camHeight = 720
camWidth = 1280
path = 'models/movenet/movenet_lightning.tflite'
interpreter = tf.lite.Interpreter(model_path=path)
connections = [
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

def preProcess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(frame, (192, 192))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    return img

def runInference(inputDetails, outputDetails, img):
    interpreter.set_tensor(inputDetails[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(outputDetails[0]['index'])

    return output

def deployMovenet(interpreter):
    interpreter.allocate_tensors()
    inputDetails = interpreter.get_input_details()
    outputDetails = interpreter.get_output_details()
    
    return (inputDetails, outputDetails)

# Switch to cv2.VideoCapture(0) or cv2.VideoCapture(1) if fails to read
cam = cv2.VideoCapture(0)
inputDetails, outputDetails = deployMovenet(interpreter)
smoothed = np.zeros((1, 17, 2))

while True:
    ret, frame = cam.read()
    
    if not ret:
        print("Failed to read")
        break

    img = preProcess(frame)
    output = runInference(inputDetails, outputDetails, img)

    # TODO: Implement One Euro filter
    confidenceThreshold = 0.3
    for i in range(17):
        point = output[0, 0, i, :]
        yPoint = point[0] * camHeight
        xPoint = point[1] * camWidth
        confidence = point[2]
        
        if confidence > confidenceThreshold:
            smoothed[0, i, 0] = (xPoint * a) + smoothed[0, i, 0] * (1 - a)
            smoothed[0, i, 1] = (yPoint * a) + smoothed[0, i, 1] * (1 - a)

    # Draw keypoints and connections
    for (start, end) in connections:
        cv2.line(frame, (int(smoothed[0, start, 0]), int(smoothed[0, start, 1])), (int(smoothed[0, end, 0]), int(smoothed[0, end, 1])), (255, 0, 0), 3)
    for i in range(17):
        cv2.circle(frame, (int(smoothed[0, i, 0]), int(smoothed[0, i, 1])), 8, (255, 0, 0), -1)

    cv2.imshow('MacBook Camera', frame)

    if cv2.waitKey(1) == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()