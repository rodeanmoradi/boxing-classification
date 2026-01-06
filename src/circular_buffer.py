import numpy as np

class Buffer:
    def __init__(self):
        self.arr = np.zeros((30, 17, 2))
        self.index = 0    
        self.n = 30  
        self.count = 0
    
    def normalize(self, smoothed):
        hips_midpoint = (smoothed[11] + smoothed[12]) / 2
        hip_width = np.linalg.norm(smoothed[12] - smoothed[11])
        normalized = (smoothed - hips_midpoint) / (hip_width)

        return normalized

    def fill_buffer(self, smoothed):
        normalized = self.normalize(smoothed)
        self.arr[self.index] = normalized
        self.index = (self.index + 1) % self.n

        if self.count < self.n:
            self.count += 1
    
    def order_buffer(self):
        
        if self.count < self.n:
            return None
        
        ordered = np.roll(self.arr, -self.index, axis=0)

        return ordered.reshape(1, 30, -1)