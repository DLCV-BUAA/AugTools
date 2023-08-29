from augtools.img.transform import DualTransform
import augtools.img.functional as F


class VerticalFlip(DualTransform):

    def _append_extensions(self):
        from augtools.extensions.get_image_param_extension import GetImageParamExtension
        return [GetImageParamExtension()]

    def _compute_x_function(self, img, rs=None):
        return F.vflip(img)

    def _compute_bbox_function(self, y, rs=None):
        h, w, bboxes = rs['rows'], rs['cols'], []
        for item in y:
            bboxes.append(F.bbox_vflip(item, h, w))
        return bboxes

    def _compute_keypoint_function(self, y, rs=None):
        h, w, points = rs['rows'], rs['cols'], []
        for item in y:
            points.append(F.keypoint_vflip(item, h, w))
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
#     bbox = [(170, 30, 300, 220)] # 500 375 500 375
#     keypoint = [(230, 80, 1, 1)]
#
#     show_bbox_keypoint_image(img, bbox=bbox, keypoint=keypoint)
#
#     transform = VerticalFlip()
#     re = transform(img=img, force_apply=True, bbox=bbox, keypoint=keypoint)
#     show_bbox_keypoint_image(re['img'], bbox=re['bbox'], keypoint=re['keypoint'])
#
#     cc = A.VerticalFlip(always_apply=True)
#     result = cc(image=img, bboxes=bbox, keypoints=keypoint)
#     show_bbox_keypoint_image_float(result['image'], bbox=result['bboxes'], keypoint=result['keypoints'])
