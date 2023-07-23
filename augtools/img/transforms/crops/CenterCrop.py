from augtools.img.transform import DualTransform
from augtools.img.transforms.utils.img_utils import *
import skimage as sk
import random
from typing_extensions import Concatenate, ParamSpec
import math
import functional as F


class CenterCrop(DualTransform):
    """Crop the central part of the input.

    Args:
        height (int): height of the crop.
        width (int): width of the crop.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32

    Note:
        It is recommended to use uint8 images as input.
        Otherwise the operation will require internal conversion
        float32 -> uint8 -> float32 that causes worse performance.
    """

    def __init__(self, height, width, always_apply=False, p=1.0):
        super(CenterCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def __call__(self, *args, force_apply: bool = False, **kwargs):
        if (random.random() < self.p) and not self.always_apply and not force_apply:
            return kwargs
        res = {}

        h, w, _ = kwargs["img"].shape
        if "img" in kwargs:
            res["img"] = F.center_crop(kwargs["img"], self.height, self.width)

        if "bbox" in kwargs:
            res["bbox"] = F.bbox_center_crop(kwargs["bbox"], self.height, self.width, h, w)

        if "keypoint" in kwargs:
            res["keypoint"] = F.keypoint_center_crop(kwargs["keypoint"], self.height, self.width, h, w)

        return res


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    # print(img)
    bbox = (50, 60, 50, 80)

    transform = CenterCrop(100, 100)
    result = transform(img=img, force_apply=True, bbox=bbox)
    # print(result['img'])
    print(result["bbox"])
    show_image(result['img'])