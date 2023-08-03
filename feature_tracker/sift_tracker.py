import numpy as np
import cv2


class SiftTracker():
    """
        using sift+flann to track features
    """
    def __init__(self, min_match_count=4):
        # set detector 
        self.detector = cv2.SIFT_create()

        # set flann
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        self.min_match_count = min_match_count

    def initialize(self, *args, **kwargs):
        pass
    
    def track_features(self, src, dst, *args, **kwargs):
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        
        kp_src, des_src = self.detector.detectAndCompute(src_gray, None)
        kp_dst, des_dst = self.detector.detectAndCompute(dst_gray, None)

        matches = self.flann.knnMatch(des_src, des_dst, k=2)
        min_distance = 0xfffffff
        for m, n in matches:
            if m.distance > 0.7*n.distance:
                continue
            if m.distance < min_distance:
                min_distance = m.distance
        min_distance = 5 * max(min_distance, 10.0)
        good = []
        for m, n in matches:
            # if m.distance < 0.7*n.distance and m.distance < min_distance:
            if m.distance < 0.7*n.distance:
                good.append(m)

        pts_src = np.float32([kp_src[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_dst = np.float32([kp_dst[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        return pts_src, pts_dst

