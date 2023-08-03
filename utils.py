import json
import logging
import os
import shutil
import numpy as np
import torch


def getAllFileInDir(fileDir):
    path = []
    for root, dirs, files in os.walk(fileDir):
        if len(dirs) != 0:
            continue
        for fileName in files:
            path.append(os.path.join(root, fileName))
    return path

def checkAndMakeDir(dir):
    if not os.path.exists(dir):
        print("Directory does not exist! Making directory {}".format(dir))
        os.makedirs(dir)

def getMeshGrid(frame_width, frame_height):
    warp_grid_x, warp_grid_y = np.meshgrid(np.linspace(0, frame_width-1, frame_width),
                                           np.linspace(0, frame_height-1, frame_height))
    return warp_grid_x, warp_grid_y

def getBaseGrid(frame_width, frame_height):
    warp_grid_x, warp_grid_y = getMeshGrid(frame_width, frame_height)   # (H, W)
    base_grid = np.stack((warp_grid_x, warp_grid_y), axis=-1).astype(np.float32)    # (H, W, 2)
    return base_grid

def mulScalar(tensor, x_scalar, y_scalar):
    tensor[:, 0, :, :] *= x_scalar
    tensor[:, 1, :, :] *= y_scalar

def homoMul(mat0, mat1):
    result = np.matmul(mat0, mat1)
    result = result / result[2, 2]
    return result

def homoInv(mat):
    mat = np.linalg.inv(mat)
    mat = mat / mat[2, 2]
    return mat

def concatImagesHorizon(img_list):
    new_img = img_list[0].copy()
    for i in range(1, len(img_list)):
        new_img = np.concatenate((new_img, img_list[i]), axis=1)
    return new_img

def concatImagesVertical(img_list):
    new_img = img_list[0].copy()
    for i in range(1, len(img_list)):
        new_img = np.concatenate((new_img, img_list[i]), axis=0)
    return new_img

class Params():

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():

    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
   
    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
   
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
   
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def flow_to_image(flow, display=False):
   
    def compute_color(u, v):
        def make_color_wheel():
            RY = 15
            YG = 6
            GC = 4
            CB = 11
            BM = 13
            MR = 6

            ncols = RY + YG + GC + CB + BM + MR

            colorwheel = np.zeros([ncols, 3])

            col = 0

            # RY
            colorwheel[0:RY, 0] = 255
            colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
            col += RY

            # YG
            colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
            colorwheel[col:col + YG, 1] = 255
            col += YG

            # GC
            colorwheel[col:col + GC, 1] = 255
            colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
            col += GC

            # CB
            colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
            colorwheel[col:col + CB, 2] = 255
            col += CB

            # BM
            colorwheel[col:col + BM, 2] = 255
            colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
            col += +BM

            # MR
            colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
            colorwheel[col:col + MR, 0] = 255

            return colorwheel
        
        [h, w] = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0

        colorwheel = make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u**2 + v**2)

        a = np.arctan2(-v, -u) / np.pi

        fk = (a + 1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)

        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel, 1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx = np.logical_not(idx)

            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

        return img

    UNKNOWN_FLOW_THRESH = 1e7
  
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu, maxu, minv, maxv))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    _min, _mean, _max = np.min(flow), np.mean(flow), np.max(flow)

    return np.uint8(img)
