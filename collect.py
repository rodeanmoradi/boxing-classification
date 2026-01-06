import cv2
import time
import numpy as np
from src import PoseEstimator, SmoothingFilter, Visualiser, Buffer

def main():

    cam = cv2.VideoCapture(1) # Pass 0 or 1
    frame_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    prev_time = time.time()
    jab_count = 0
    cross_count = 0
    hook_count = 0
    uppercut_count = 0

    movenet_lightning = PoseEstimator('models/movenet/movenet_lightning.tflite')
    one_euro = SmoothingFilter()
    visualiser = Visualiser()
    circular_buffer = Buffer()
    
    while True:
        current_time = time.time()
        frame_time = current_time - prev_time
        prev_time = current_time
        key = cv2.waitKey(1)
        
        ret, frame = cam.read()
        
        if not ret:
            print("Failed to read")
            break

        smoothed = one_euro.filter(movenet_lightning.detect(frame), frame_height, frame_width, frame_time)
        visualiser.draw_keypoints(frame, smoothed)
        circular_buffer.fill_buffer(smoothed)

        if key == ord('j'):
            time.sleep(5)
            np.save(f'data/raw/jab/jab_{jab_count}.npy', circular_buffer.order_buffer())
            jab_count += 1

        elif key == ord('c'):
            time.sleep(5)
            np.save(f'data/raw/cross/cross_{cross_count}.npy', circular_buffer.order_buffer())
            cross_count += 1

        elif key == ord('h'):
            time.sleep(5)
            np.save(f'data/raw/hook/hook_{hook_count}.npy', circular_buffer.order_buffer())
            hook_count += 1
            
        elif key == ord('u'):
            time.sleep(5)
            np.save(f'data/raw/uppercut/uppercut_{uppercut_count}.npy', circular_buffer.order_buffer())
            uppercut_count += 1

        cv2.imshow('MacBook Camera', frame)

        if key == ord('x'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()