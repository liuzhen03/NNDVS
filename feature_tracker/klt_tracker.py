import numpy as np
import cv2

class KltTracker():
    """
        using fast+klt to track features
    """
    def __init__(self, back_threshold=1.0, nms=True, check_optical_flow=False):
      
        # set detector
        self.detector = cv2.FastFeatureDetector_create()
        # disable nonmaxSuppression
        if not nms:
            self.detector.setNonmaxSuppression(0)

        # set check threshold
        self.back_threshold = back_threshold

        self.check_optical_flow = check_optical_flow
    
    def initialize(self, *args, **kwargs):
        pass

    def check_trace(self, img0, img1, p0):
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, winSize=(17, 17), maxLevel=4,
                                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))
        if not self.check_optical_flow:
            return p1, _st
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, winSize=(17, 17), maxLevel=4,
                                                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))
        d = abs(p0-p0r).reshape(-1, 2).max(-1)
        status = d < self.back_threshold
        return p1, status
    
    def track_features(self, src, dst, *args, **kwargs):
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        kp_src = self.detector.detect(src_gray, None)
        kp_src = np.float32([kp.pt for kp in kp_src]).reshape(-1, 1, 2)
        kp_dst, st= self.check_trace(src_gray, dst_gray, kp_src)

        # Select good points
        if kp_dst is not None:
            pts_src = kp_src[st==1]
            pts_dst = kp_dst[st==1]
        
        return pts_src, pts_dst