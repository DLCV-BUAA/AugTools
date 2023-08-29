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

    def __init__(self, always_apply: bool = False, p: float = 0.5, return_type=None):
        super(Flip, self).__init__(always_apply, p, return_type)
        self.d = random.randint(-1, 1)

    def _append_extensions(self):
        from augtools.extensions.get_image_param_extension import GetImageParamExtension
        return [GetImageParamExtension()]

    def _compute_x_function(self, img, rs=None):
        return F.random_flip(img, self.d)

    def _compute_bbox_function(self, y, rs=None):
        h, w, bboxes = rs['rows'], rs['cols'], []
        for item in y:
            bboxes.append(F.bbox_flip(item, self.d, h, w))
        return bboxes

    def _compute_keypoint_function(self, y, rs=None):
        h, w, points = rs['rows'], rs['cols'], []
        for item in y:
            points.append(F.keypoint_flip(item, self.d, h, w))
        return points

# if __name__ == '__main__':
#     from augtools.utils.test_utils import *
#
#     prefix = f'../test/'
#     image = prefix + 'test.jpg'
#
#     img = read_image(image)
#     print(img.shape)
#     bbox = [(170, 30, 300, 220)]
#     keypoint = [(230, 80, 1, 1)]
#
#     show_bbox_keypoint_image(img, bbox=bbox, keypoint=keypoint)
#
#     transform = Flip()
#     re = transform(img=img, force_apply=True, bbox=bbox, keypoint=keypoint)
#     show_bbox_keypoint_image(re['img'], bbox=re['bbox'], keypoint=re['keypoint'])
#
#     # cc = A.HorizontalFlip(always_apply=True)
#     # result = cc(image=img, bboxes=bbox, keypoints=keypoint)
#     # show_bbox_keypoint_image(result['image'], bbox=result['bboxes'], keypoint=result['keypoints'])
