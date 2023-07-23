import random

from augtools.img.transform import DualTransform
import functional as F

class Flip(DualTransform):
    """Flip the input either horizontally, vertically or both horizontally and vertically.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def _compute_x_function(self, img, rs=None):
        """Args:
        d (int): code that specifies how to flip the input. 0 for vertical flipping, 1 for horizontal flipping,
                -1 for both vertical and horizontal flipping (which is also could be seen as rotating the input by
                180 degrees).
        """
        d = random.randint(0, 1)
        return F.random_flip(img, d)

    # def get_params(self):
    #     # Random int in the range [-1, 1]
    #     return {"d": random.randint(-1, 1)}
    #
    # def _compute_bbox_function(self, y, rs=None):
    #     return F.bbox_flip(y)

    # def _compute_keypoint_function(self, y, rs=None):
    #   return F.keypoint_flip(keypoint, **params)


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    # bboxs = [(50, 60, 50, 80), (50, 60, 50, 80)]
    bbox = (50, 60, 50, 80)

    transform = Flip()
    result = transform(img=img, force_apply=True)
    print(result['img'])
    # print(result['bbox'])

    show_image(result['img'])