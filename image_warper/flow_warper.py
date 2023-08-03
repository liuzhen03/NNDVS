import numpy as np
import cv2

from utils import getBaseGrid


class FlowWarper():

    def __init__(self):
      
        self.output_width = 1
        self.output_height = 1
        self.base_grid = None

    def initialize(self, output_width, output_height, *args, **kwargs):
        self.output_width = output_width
        self.output_height = output_height
        self.base_grid = getBaseGrid(output_width, output_height)

    def warp_image(self, img, trans, *args, **kwargs):
        trans = -trans + self.base_grid
        new_img = cv2.remap(img, trans[:, :, 0], trans[:, :, 1], cv2.INTER_LINEAR)
        return new_img
