import numpy as np

PI = 3.14

class SmoothingFilter:

    def __init__(self):
        self.confidence_threshold = 0.3 
        self.b = 0.1 
        self.f_c_min = 1.2 
        self.f_c_d = 1.0 
        self.prev_points = np.zeros((17, 2))
        self.prev_speed = np.zeros((17, 2))
        self.smoothed = np.zeros((17, 2))

    def filter(self, output, frame_height, frame_width, sampling_period):
        
        if sampling_period <= 0.00001: 
            sampling_period = 0.016
        
        coords = output[0, 0, :, :2].copy()
        coords = coords[:, [1, 0]]
        coords = coords * np.array([frame_width, frame_height])

        if np.all(self.prev_points == 0):
            self.prev_points = coords.copy()
            self.smoothed = coords.copy()

        confidence = output[0, 0, :, 2].copy()
        mask = confidence > self.confidence_threshold
        mask = mask[:, np.newaxis]

        x_dot = (coords - self.prev_points) / sampling_period
        a_d = 1 / (1 + (1 / (2 * PI * self.f_c_d * sampling_period)))
        x_dot_hat = x_dot * a_d + self.prev_speed * (1 - a_d)

        f_c = self.f_c_min + self.b * abs(x_dot_hat)
        a = 1 / (1 + (1 / (2 * PI * f_c * sampling_period)))
        x_hat = coords * a + self.smoothed * (1 - a)

        self.smoothed = np.where(mask, x_hat, self.smoothed)
        self.prev_speed = np.where(mask, x_dot_hat, self.prev_speed)
        self.prev_points = np.where(mask, coords, self.prev_points)
        
        return self.smoothed