from augtools.img.transform import DualTransform
import random
import augtools.img.functional as F
import albumentations as A


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
            bboxes = []
            for item in kwargs["bbox"]:
                bboxes.append(F.bbox_center_crop(item, self.height, self.width, h, w))
            res["bbox"] = bboxes

        if "keypoint" in kwargs:
            points = []
            for item in kwargs["keypoint"]:
                points.append(F.keypoint_center_crop(item, self.height, self.width, h, w))
            res["keypoint"] = points

        return res


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    bbox = [(30, 170, 230, 300)]
    keypoint = [(230, 80, 1, 1)]

    transform = CenterCrop(300, 350)
    re = transform(img=img, force_apply=True, bbox=bbox, keypoint=keypoint)
    show_bbox_keypoint_image(img, bbox=bbox, keypoint=keypoint)

    cc = A.CenterCrop(300, 350, always_apply=True)
    result = cc(image=img, bboxes=bbox, keypoints=keypoint)
    show_bbox_keypoint_image(re['img'], bbox=re['bbox'], keypoint=re['keypoint'])
    show_bbox_keypoint_image(result['image'], bbox=result['bboxes'], keypoint=result['keypoints'])
