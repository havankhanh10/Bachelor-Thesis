import numpy as np
kinect = False
if not kinect:
    zed = True
else:
    zed = False

class camera_params:
    if kinect:
        width = 4096
        height = 3072
        fx = 1957.02
        fy = 1957.01
        cx = 2058.15
        cy = 1540.8
    elif zed:
        width = 1920
        height = 1080
        fx = 1047.9615478515625
        fy = 1047.9615478515625
        cx = 972.1301879882812
        cy = 558.8695678710938

    def get_camera_matrix(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])