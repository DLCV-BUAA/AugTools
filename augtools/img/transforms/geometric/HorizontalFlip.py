import warnings
import random
from itertools import product

import cv2
import numpy as np

from augtools.img.transform import DualTransform
import functional as F


class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def _compute_x_function(self, img, rs=None):
        return F.hflip(img)

    def _compute_bbox_function(self, y, rs=None):
        return F.bbox_hflip(y)

    # def _compute_keypoint_function(self, y, rs=None):
    #    return F.keypoint_hflip(keypoint, **params)




if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    # bboxs = [(50, 60, 50, 80), (50, 60, 50, 80)]
    bbox = (50, 60, 50, 80)

    transform = HorizontalFlip()
    result = transform(img=img, bbox=bbox, force_apply=True)
    print(result['img'])

    show_image(result['img'])
