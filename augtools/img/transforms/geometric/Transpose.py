from augtools.img.transform import DualTransform
import augtools.img.functional as F


class Transpose(DualTransform):

    def _append_extensions(self):
        from augtools.extensions.get_image_param_extension import GetImageParamExtension
        return [GetImageParamExtension()]

    def _compute_x_function(self, img, rs=None):
        return F.transpose(img)

    def _compute_bbox_function(self, y, rs=None):
        h, w, bboxes = rs['rows'], rs['cols'], []
        for item in y:
            bboxes.append(F.bbox_transpose(item, 0))
        return bboxes

    def _compute_keypoint_function(self, y, rs=None):
        h, w, points = rs['rows'], rs['cols'], []
        for item in y:
            points.append(F.keypoint_transpose(item))
        return points

# if __name__ == '__main__':
#     from augtools.utils.test_utils import *
#     import albumentations as A
#
#     prefix = f'../test/'
#     image = prefix + 'test.jpg'
#
#     img = read_image(image)
#     print(img.shape)
#     bbox = [(170 / 500, 30 / 375, 300 / 500, 220 / 375)]
#     keypoint = [(230, 80, 1, 1)]
#
#     show_bbox_keypoint_image_float(img, bbox=bbox, keypoint=keypoint)
#
#     transform = Transpose()
#     re = transform(img=img, force_apply=True, bbox=bbox, keypoint=keypoint)
#     show_bbox_keypoint_image_float(re['img'], bbox=re['bbox'], keypoint=re['keypoint'])
#
#     cc = A.Transpose(always_apply=True)
#     result = cc(image=img, bboxes=bbox, keypoints=keypoint)
#     show_bbox_keypoint_image_float(result['image'], bbox=result['bboxes'], keypoint=result['keypoints'])
