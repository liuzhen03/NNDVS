import numpy as np
import cv2


class GlobalWarper():

    def __init__(self):

        self.output_width = 1
        self.output_height = 1

    def initialize(self, output_width, output_height, *args, **kwargs):
        self.output_width = output_width
        self.output_height = output_height

    def warp_image(self, img, trans, *args, **kwargs):
        new_img = cv2.warpPerspective(img, trans,
                                     (self.output_width, self.output_height))
        return new_img
