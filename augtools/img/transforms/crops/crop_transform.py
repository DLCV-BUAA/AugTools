from augtools.img.transform import DualTransform
import random
import augtools.img.functional as F
import albumentations as A


class Crop(DualTransform):
    """Crop region from image.

    Args:
        x_min (int): Minimum upper left x coordinate.
        y_min (int): Minimum upper left y coordinate.
        x_max (int): Maximum lower right x coordinate.
        y_max (int): Maximum lower right y coordinate.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, x_min=0, y_min=0, x_max=1024, y_max=1024, always_apply=False, p=1.0):
        super(Crop, self).__init__(always_apply, p)
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

    def __call__(self, *args, force_apply: bool = False, **kwargs):
        if (random.random() < self.p) and not self.always_apply and not force_apply:
            return kwargs
        res = {}

        h, w, _ = kwargs["img"].shape
        if "img" in kwargs:
            res["img"] = F.crop(kwargs["img"], x_min=self.x_min, y_min=self.y_min, x_max=self.x_max, y_max=self.y_max)

        if "bbox" in kwargs:
            bboxes = []
            for item in kwargs["bbox"]:
                bboxes.append(F.bbox_crop(item, x_min=self.x_min, y_min=self.y_min, x_max=self.x_max, y_max=self.y_max,
                                          rows=h, cols=w))
            res["bbox"] = bboxes

        if "keypoint" in kwargs:
            points = []
            for item in kwargs["keypoint"]:
                points.append(F.crop_keypoint_by_coords(item,
                                                        crop_coords=(self.x_min, self.y_min, self.x_max, self.y_max)))
            res["keypoint"] = points
        return res


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    bbox = [(30, 170, 230, 300)]

    transform = Crop(20, 240, 180, 310)
    re = transform(img=img, force_apply=True, bbox=bbox)
    show_bbox_keypoint_image(img, bbox=bbox, keypoint=None)

    cc = A.Crop(20, 240, 180, 310, always_apply=True)
    result = cc(image=img, bboxes=bbox)
    show_bbox_keypoint_image(re['img'], bbox=re['bbox'], keypoint=None)
    show_bbox_keypoint_image(result['image'], bbox=result['bboxes'], keypoint=None)
