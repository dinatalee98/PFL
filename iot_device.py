import numpy as np

class IoTDevice:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.comp_time = t
    
    def get_location(self):
        return np.array([self.x, self.y])
    
    def get_computation_time(self):
        return self.comp_time