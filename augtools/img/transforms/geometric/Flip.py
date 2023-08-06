import random
from augtools.img.transform import DualTransform
import augtools.img.functional as F


class Flip(DualTransform):
    """Flip the input either horizontally, vertically or both horizontally and vertically.

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

        d = random.randint(-1, 1)
        h, w, _ = kwargs["img"].shape

        if "img" in kwargs:
            res["img"] = F.random_flip(kwargs["img"], d)

        if "bbox" in kwargs:
            bboxes = []
            for item in kwargs["bbox"]:
                bboxes.append(F.bbox_flip(item, d, h, w))
            res["bbox"] = bboxes

        if "keypoint" in kwargs:
            points = []
            for item in kwargs["keypoint"]:
                points.append(F.keypoint_flip(item, d, h, w))
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

    transform = Flip()
    re = transform(img=img, force_apply=True, bbox=bbox, keypoint=keypoint)
    show_bbox_keypoint_image(re['img'], bbox=re['bbox'], keypoint=re['keypoint'])

    # cc = A.HorizontalFlip(always_apply=True)
    # result = cc(image=img, bboxes=bbox, keypoints=keypoint)
    # show_bbox_keypoint_image(result['image'], bbox=result['bboxes'], keypoint=result['keypoints'])
