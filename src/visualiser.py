import cv2

class Visualiser:
    
    def __init__(self):
        self.connections = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (5, 11), (6, 8), (6, 12), (7, 9),
            (8, 10), (11, 13), (12, 14), (13, 15), (14, 16)]
        
    def draw_keypoints(self, frame, smoothed):
        for (start, end) in self.connections:
            cv2.line(frame, (int(smoothed[start, 0]), int(smoothed[start, 1])), 
                     (int(smoothed[end, 0]), int(smoothed[end, 1])), (255, 0, 0), 3)
        for i in range(17):
            cv2.circle(frame, (int(smoothed[i, 0]), int(smoothed[i, 1])), 8, (255, 0, 0), -1)