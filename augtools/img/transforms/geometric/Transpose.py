from augtools.img.transform import DualTransform
import random
import augtools.img.functional as F
import albumentations as A


class Transpose(DualTransform):
    """Transpose the input by swapping rows and columns.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __call__(self, *args, force_apply: bool = False, **kwargs):
        if (random.random() < self.p) and not self.always_apply and not force_apply:
            return kwargs
        res = {}

        if "img" in kwargs:
            res["img"] = F.transpose(kwargs["img"])

        if "bbox" in kwargs:
            bboxes = []
            for item in kwargs["bbox"]:
                bboxes.append(F.bbox_transpose(item, 0))
            res["bbox"] = bboxes

        if "keypoint" in kwargs:
            points = []
            for item in kwargs["keypoint"]:
                points.append(F.keypoint_transpose(item))
            res["keypoint"] = points

        return res


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    print(img.shape)
    bbox = [(170 / 500, 30 / 375, 300 / 500, 220 / 375)]
    keypoint = [(230, 80, 1, 1)]

    show_bbox_keypoint_image_float(img, bbox=bbox, keypoint=keypoint)

    transform = Transpose()
    re = transform(img=img, force_apply=True, bbox=bbox, keypoint=keypoint)
    show_bbox_keypoint_image_float(re['img'], bbox=re['bbox'], keypoint=re['keypoint'])

    cc = A.Transpose(always_apply=True)
    result = cc(image=img, bboxes=bbox, keypoints=keypoint)
    show_bbox_keypoint_image_float(result['image'], bbox=result['bboxes'], keypoint=result['keypoints'])
