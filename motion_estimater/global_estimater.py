import numpy as np
import cv2


class GlobalEstimater():
    """
        estimate a global homography matrix as motion
    """
    def __init__(self):
        pass

    def initialize(self, *args, **kwargs):
        pass
    
    def estimate_motion(self, pts_src, pts_dst, *args, **kwargs):
        # estimate a 3x3 homography matrix
        homo = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC)
        return homo[0]
