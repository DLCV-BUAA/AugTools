from augtools.img.transform import DualTransform
from augtools.img.transforms.utils.img_utils import *
import random
import albumentations as A


class Perspective(DualTransform):
    """Perform a random four point perspective transform of the input.

    Args:
        scale (float or (float, float)): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners.
            If scale is a single float value, the range will be (0, scale). Default: (0.05, 0.1).
        keep_size (bool): Whether to resize image’s back to their original size after applying the perspective
            transform. If set to False, the resulting images may end up having different shapes
            and will always be a list, never an array. Default: True
        pad_mode (OpenCV flag): OpenCV border mode.
        pad_val (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
            Default: 0
        mask_pad_val (int, float, list of int, list of float): padding value for mask
            if border_mode is cv2.BORDER_CONSTANT. Default: 0
        fit_output (bool): If True, the image plane size and position will be adjusted to still capture
            the whole image after perspective transformation. (Followed by image resizing if keep_size is set to True.)
            Otherwise, parts of the transformed image may be outside of the image plane.
            This setting should not be set to True when using large scale values as it could lead to very large images.
            Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32
    """

    def __init__(
            self,
            scale=(0.05, 0.1),
            keep_size=True,
            pad_mode=cv2.BORDER_CONSTANT,
            pad_val=0,
            mask_pad_val=0,
            fit_output=False,
            interpolation=cv2.INTER_LINEAR,
            always_apply=False,
            p=0.5,
    ):
        super().__init__(always_apply, p)
        self.scale = to_tuple(scale, 0)
        self.keep_size = keep_size
        self.pad_mode = pad_mode
        self.pad_val = pad_val
        self.mask_pad_val = mask_pad_val
        self.fit_output = fit_output
        self.interpolation = interpolation
        self.pers = A.Perspective(scale, keep_size, pad_mode, pad_val, mask_pad_val, fit_output, interpolation,
                                  interpolation)

    def __call__(self, *args, force_apply: bool = False, **kwargs):
        if (random.random() < self.p) and not self.always_apply and not force_apply:
            return kwargs
        else:
            return self.pers(**kwargs, force_apply=force_apply)


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    print(img.shape)
    bbox = [(170 / 500, 30 / 375, 300 / 500, 220 / 375)]
    keypoint = [(230, 80, 1, 1)]

    show_bbox_keypoint_image_float(img, bbox=bbox, keypoint=keypoint)

    transform = Perspective()
    re = transform(image=img, force_apply=True, bboxes=bbox, keypoints=keypoint)
    show_bbox_keypoint_image_float(re['image'], bbox=re['bboxes'], keypoint=re['keypoints'])

    cc = A.Perspective(always_apply=True)
    result = cc(image=img, bboxes=bbox, keypoints=keypoint)
    show_bbox_keypoint_image_float(result['image'], bbox=result['bboxes'], keypoint=result['keypoints'])
