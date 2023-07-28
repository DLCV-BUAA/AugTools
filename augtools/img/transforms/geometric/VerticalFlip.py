import warnings
import random
from itertools import product

import cv2
import numpy as np

from augtools.img.transform import DualTransform
import functional as F

# DualTransform 是对 y 也需要处理
# BasicTrans **kwargs 传入，然后 BasicTrans _compute_x/y 传入是 **kwargs
#

class VerticalFlip(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def _compute_x_function(self, img, rs=None):
        return F.vflip(img)

    def _compute_bbox_function(self, y, rs=None):
        return F.bbox_vflip(y)

    # def _compute_keypoint_function(self, y, rs=None):
    #     return F.keypoint_vflip(y, **params)


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    # bboxs = [(50, 60, 50, 80), (50, 60, 50, 80)]
    bbox = (50, 60, 50, 80)

    transform = VerticalFlip()
    result = transform(img=img, bbox=bbox, force_apply=True)
    print(result['img'])
    print(result['bbox'])

    show_image(result['img'])
