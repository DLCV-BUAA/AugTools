from augtools.img.transform import DualTransform
import albumentations as A
import random
import augtools.img.functional as F


class HorizontalFlip(DualTransform):
    """Flip the input horizontally around the y-axis.

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
        h, w, _ = kwargs["img"].shape

        if "img" in kwargs:
            res["img"] = F.hflip(img)

        if "bbox" in kwargs:
            bboxes = []
            for item in kwargs["bbox"]:
                bboxes.append(F.bbox_hflip(item, h, w))
            res["bbox"] = bboxes

        if "keypoint" in kwargs:
            points = []
            for item in kwargs["keypoint"]:
                points.append(F.keypoint_hflip(item, h, w))
            res["keypoint"] = points

        return res


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    print(img.shape)
    bbox = [(170, 30, 300, 220)]
    keypoint = [(230, 80, 1, 1)]

    show_bbox_keypoint_image(img, bbox=bbox, keypoint=keypoint)

    transform = HorizontalFlip()
    re = transform(img=img, force_apply=True, bbox=bbox, keypoint=keypoint)
    show_bbox_keypoint_image(re['img'], bbox=re['bbox'], keypoint=re['keypoint'])

    cc = A.HorizontalFlip(always_apply=True)
    result = cc(image=img, bboxes=bbox, keypoints=keypoint)
    show_bbox_keypoint_image(result['image'], bbox=result['bboxes'], keypoint=result['keypoints'])
