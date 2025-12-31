import cv2
import time

from src import PoseEstimator, SmoothingFilter, Visualiser, Buffer

cam = cv2.VideoCapture(1) # Pass 0 or 1
frame_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
prev_time = time.time()

movenet_lightning = PoseEstimator('models/movenet/movenet_lightning.tflite')
one_euro = SmoothingFilter()
visualiser = Visualiser()
circular_buffer = Buffer()

while True:
    current_time = time.time()
    frame_time = current_time - prev_time
    prev_time = current_time
    
    ret, frame = cam.read()
    
    if not ret:
        print("Failed to read")
        break

    smoothed = one_euro.filter(movenet_lightning.detect(frame), frame_height, frame_width, frame_time)

    visualiser.draw_keypoints(frame, smoothed)

    circular_buffer.fill_buffer(smoothed)

    cv2.imshow('MacBook Camera', frame)

    if cv2.waitKey(1) == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()