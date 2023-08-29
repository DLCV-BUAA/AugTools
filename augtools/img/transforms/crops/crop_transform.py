from augtools.img.transform import DualTransform
import augtools.img.functional as F
from augtools.extensions.get_image_param_extension import GetImageParamExtension


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

    def _append_extensions(self):
        return [GetImageParamExtension()]

    def _compute_x_function(self, img, rs=None):
        return F.crop(img, x_min=self.x_min, y_min=self.y_min, x_max=self.x_max, y_max=self.y_max)

    def _compute_bbox_function(self, y, rs=None):
        h, w, bboxes = rs['rows'], rs['cols'], []
        for item in y:
            bboxes.append(F.bbox_crop(item, x_min=self.x_min, y_min=self.y_min, x_max=self.x_max, y_max=self.y_max,
                                      rows=h, cols=w))
        return bboxes

    def _compute_keypoint_function(self, y, rs=None):
        h, w, points = rs['rows'], rs['cols'], []
        for item in y:
            points.append(F.crop_keypoint_by_coords(item, crop_coords=(self.x_min, self.y_min, self.x_max, self.y_max)))
        return points

# if __name__ == '__main__':
#     from augtools.utils.test_utils import *
#     import albumentations as A
#
#     prefix = f'../test/'
#     image = prefix + 'test.jpg'
#
#     img = read_image(image)
#     bbox = [(30, 170, 230, 300)]
#
#     transform = Crop(20, 240, 180, 310)
#     re = transform(img=img, force_apply=True, bbox=bbox)
#     show_bbox_keypoint_image(img, bbox=bbox, keypoint=None)
#
#     cc = A.Crop(20, 240, 180, 310, always_apply=True)
#     result = cc(image=img, bboxes=bbox)
#     show_bbox_keypoint_image(re['img'], bbox=re['bbox'], keypoint=None)
#     show_bbox_keypoint_image(result['image'], bbox=result['bboxes'], keypoint=None)
