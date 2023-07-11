import warnings
import random
from itertools import product

import cv2
import numpy as np

from augtools.img.transform import DualTransform
from augtools.img.transforms.utils.img_utils import *
from augtools.img.transforms.utils.bbox_utils import *
from augtools.img.transforms.utils.keypoint_utils import *
from augtools.extensions.get_image_param_extension import GetImageParamExtension


class RandomCrop(DualTransform):
    """Crop a random part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, height, width, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width
        
        self.h_start = random.random()
        self.w_start = random.random()

    def _append_extensions(self):
        return [GetImageParamExtension()]
    
    def _compute_x_function(self, img, rs=None):
        height, width = img.shape[:2]
        if height < self.height or width < self.width:
            raise ValueError(
                "Requested crop size ({crop_height}, {crop_width}) is "
                "larger than the image size ({height}, {width})".format(
                    crop_height=self.height, crop_width=self.width, height=height, width=width
                )
            )
        x1, y1, x2, y2 = get_random_crop_coords(height, width, self.height, self.width, self.h_start, self.w_start)
        img = img[y1:y2, x1:x2]
        return img

    
    @lists_process
    def _compute_bbox_function(self, y, rs=None):
        crop_coords = get_random_crop_coords(rs['rows'], rs['cols'], self.height, self.width, self.h_start, self.w_start)
        return crop_bbox_by_coords(y, crop_coords, self.height, self.width, rs['rows'], rs['cols'])


    @lists_process
    def _compute_keypoint_function(self, y, rs=None):
        crop_coords = get_random_crop_coords(rs['rows'], rs['cols'], self.height, self.width, self.h_start, self.w_start)
        return crop_keypoint_by_coords(y, crop_coords)
    
    
if __name__ == '__main__':
    from augtools.utils.test_utils import *
    from augtools.core.compose import Sequential
    from augtools.img.transforms.blur.blur_transform import Blur
    from augtools.img.transforms.blur.gaussian_blur_transform import GaussianBlur
    # prefix = '../test/'
    # image = prefix + 'test.jpg'
    image = f'/home/jiajunlong/Music/贾俊龙/数据增强/AugTools/augtools/img/transforms/test/test.jpg'
    img = read_image(image)
    # bbox = (50, 60, 50, 80)
    # keypoint = (1, 5, 3, 4)
    # print(img)
    # bboxs = [(50, 60, 50, 80), (50, 60, 50, 80)]
    
    sequential = Sequential([
        Blur(),
        GaussianBlur(),
        RandomCrop(300, 300)
    ])   
    transform = RandomCrop(100, 100)
    result = sequential(img=img, force_apply=True)

    show_image(result['img'])
    # print(result['bboxs'])
        

