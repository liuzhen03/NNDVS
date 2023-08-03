import logging
import cv2
import numpy as np
from tqdm import tqdm

from utils import *
from feature_tracker import KltTracker, SiftTracker
from motion_estimater import GlobalEstimater

class MetricAnalyzer():
    def __init__(self, frame_width, frame_height, scale_factor=40, start=31):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.scale_factor = scale_factor
        self.base_grid = getBaseGrid(frame_width, frame_height)
        self.base_grid = self.base_grid[::scale_factor, ::scale_factor, :]
        self.tracker = KltTracker()
        self.estimater = GlobalEstimater()
        self.start = 31


    def run(self, flowmap_array, video_path):
        homo_array, path_array = self.preprocess(flowmap_array, video_path)
        crop_ratio = RunningAverage()
        min_cr = 1e6
        distortion = 1e6
        for i in range(self.start, homo_array.shape[0]):
            cr = self.findAreaPersentAfterWarp(homo_array[i], self.frame_width, self.frame_height)
            min_cr = min(min_cr, cr)
            crop_ratio.update(cr)
            distortion = min(distortion, self.findAnisotropicAfterWarp(homo_array[i]))
        stab_ratio = self.findLowFrequencyPersentAfterWarp(path_array)
        
        logging.info("Crop Ratio: {:05.3f}".format(crop_ratio()))
        logging.info("Min Crop Ratio: {:05.3f}".format(min_cr))
        logging.info("Distortion Value: {:05.3f}".format(distortion))
        logging.info("Stability Score: {:05.3f}".format(stab_ratio))
        return crop_ratio(), distortion, stab_ratio
        
    def preprocess(self, flowmap_array, video_path):
        flowmap_array = flowmap_array[:, ::self.scale_factor, ::self.scale_factor, :]
        homo_array = []
        h = self.frame_height // self.scale_factor
        w = self.frame_width // self.scale_factor
        for i in range(flowmap_array.shape[0]):
            homo, _ = cv2.findHomography(self.base_grid.reshape((h*w, 1, 2)),
                                        (self.base_grid + flowmap_array[i]).reshape(h*w, 1, 2), cv2.RANSAC)
            homo_array.append(homo)

        capture = cv2.VideoCapture()
        capture.open(video_path)
        self.tracker.initialize()
        self.estimater.initialize()
        H = np.eye(3, dtype=np.float32)
        path_array = [H.copy()]
        ok, src_frame = capture.read()
        loop = tqdm(range(int(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
        for j in loop:
            ok, dst_frame = capture.read()
            if not ok:
                break
            pts_src, pts_dst = self.tracker.track_features(src_frame, dst_frame)
            if pts_src.shape[0] < 4:
                continue
            motion = self.estimater.estimate_motion(pts_src, pts_dst)
            H = np.matmul(motion, H)
            path_array.append(H.copy())
            src_frame = dst_frame
            
        homo_array = np.float32(homo_array)
        path_array = np.float32(path_array)
        return homo_array, path_array

    def adjustRectangle(self, rect, ratio):
        edge = [0, 0, 0, 0]
        edge[0] = rect[0]
        edge[1] = rect[1]
        edge[2] = rect[0] + rect[2]
        edge[3] = rect[1] + rect[3]

        edge_adj = edge.copy()
        width = edge[2] - edge[0]
        height = edge[3] - edge[1]
        center_x = (edge[2] + edge[0]) / 2
        center_y = (edge[3] + edge[1]) / 2
        r = width / height
        if r > ratio:
            adj_val = height * ratio / 2
            edge_adj[0] = center_x - adj_val
            edge_adj[2] = center_x + adj_val
        
        else:
            adj_val = width / ratio / 2
            edge_adj[1] = center_y - adj_val
            edge_adj[3] = center_y + adj_val

        rect[0] = edge_adj[0]
        rect[1] = edge_adj[1]
        rect[2] = edge_adj[2] - edge_adj[0]
        rect[3] = edge_adj[3] - edge_adj[1]
        return rect

    def findAreaPersentAfterWarp(self, homo, width, height):
        corners_src = np.array([[0, 0], [width, 0], [0, height], [width, height]], dtype=homo.dtype)
        corners_dst = cv2.perspectiveTransform(np.expand_dims(corners_src, 0), homo)
        corners_dst = np.squeeze(corners_dst, 0)

        if corners_dst[0, 0] > corners_src[0, 0]:
            corners_src[0, 0] = corners_dst[0, 0]
        if corners_dst[0, 1] > corners_src[0, 1]:
            corners_src[0, 1] = corners_dst[0, 1]

        if (corners_dst[1, 0] < corners_src[1, 0]):
            corners_src[1, 0] = corners_dst[1, 0]
        if (corners_dst[1, 1] > corners_src[1, 1]):
            corners_src[1, 1] = corners_dst[1, 1]

        if (corners_dst[2, 0] > corners_src[2, 0]):
            corners_src[2, 0] = corners_dst[2, 0]
        if (corners_dst[2, 1] < corners_src[2, 1]):
            corners_src[2, 1] = corners_dst[2, 1]

        if (corners_dst[3, 0] < corners_src[3, 0]):
            corners_src[3, 0] = corners_dst[3, 0]
        if (corners_dst[3, 1] < corners_src[3, 1]):
            corners_src[3, 1] = corners_dst[3, 1]

        rect = [0, 0, 0, 0]
        if corners_src[0, 0] > corners_src[2, 0]:
            rect[0] = corners_src[0, 0]
        else:
            rect[0] = corners_src[2, 0]
        
        if corners_src[0, 1] > corners_src[1, 1]:
            rect[1] = corners_src[0, 1]
        else:
            rect[1] = corners_src[1, 1]
        
        if corners_src[1, 0] > corners_src[3, 0]:
            rect[2] = corners_src[3, 0] - rect[0]
        else:
            rect[2] = corners_src[1, 0] - rect[0]
        
        if corners_src[3, 1] > corners_src[2, 1]:
            rect[3] = corners_src[2, 1] - rect[1]
        else:
            rect[3] = corners_src[3, 1] - rect[1]

        rect = self.adjustRectangle(rect, float(width/height))
        area = (rect[2] * rect[3]) / (width * height)
        return area

    def findAnisotropicAfterWarp(self, homo):
        affine = homo[0:2, 0:2]
        _, s, _ = np.linalg.svd(affine)
        return s[1] / s[0]

    def getCR(self, M):
        return np.sqrt(M[0,1]**2 + M[0,0]**2)

    def findLowFrequencyPersentAfterWarp(self, path_array):
        stab_ratio = RunningAverage()
        P_seq_t = []
        P_seq_r = []
        
        for i in range(self.start, path_array.shape[0]):
            Mp = path_array[i, :, :]
            transRecovered = np.sqrt(Mp[0, 2]**2 + Mp[1, 2]**2)
            thetaRecovered = np.arctan2(Mp[1, 0], Mp[0, 0]) * 180 / np.pi
            P_seq_t.append(transRecovered)
            P_seq_r.append(thetaRecovered)

        fft_t = np.fft.fft(P_seq_t)
        fft_r = np.fft.fft(P_seq_r)
        fft_t = np.abs(fft_t)**2
        fft_r = np.abs(fft_r)**2
        
        fft_t = np.delete(fft_t, 0)
        fft_r = np.delete(fft_r, 0)
        fft_t = fft_t[:len(fft_t)//2]
        fft_r = fft_r[:len(fft_r)//2]

        SS_t = np.sum(fft_t[:5])/np.sum(fft_t)
        SS_r = np.sum(fft_r[:5])/np.sum(fft_r)
        return (SS_t + SS_r) * 0.5

