import cv2
import time
import numpy as np
import torch
from src import PoseEstimator, SmoothingFilter, Visualiser, Buffer, LSTM

def main():

    cam = cv2.VideoCapture(1) # 0 or 1
    frame_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    prev_time = time.time()

    movenet_lightning = PoseEstimator('models/movenet/movenet_lightning.tflite')
    one_euro = SmoothingFilter()
    visualiser = Visualiser()
    circular_buffer = Buffer()
    model = LSTM(34, 32, 1, 2)

    # Load model
    model_dict = torch.load('models/LSTM_model.pth')
    # Set weights
    model.load_state_dict(model_dict)
    # Set model to eval
    model.eval()
    # Disable gradient calculations
    with torch.no_grad():
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
            buffer = circular_buffer.order_buffer()

            if buffer is not None:
                output = model(torch.from_numpy(buffer).float())
                # Convert logits to 0 (none) or 1 (jab)
                classification = torch.argmax(output, dim=1)
                if classification == 0:
                    cv2.putText(frame, f'No punch', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0), 3)

                if classification == 1:
                    cv2.putText(frame, f'Jab detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 0, 0), 3)

            cv2.imshow('MacBook Camera', frame)

            if cv2.waitKey(1) == ord('x'):
                break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()