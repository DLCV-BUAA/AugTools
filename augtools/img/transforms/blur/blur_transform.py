import random
from augtools.img.transform import ImageTransform
from augtools.img.transforms.utils.img_utils import *


class Blur(ImageTransform):
    """Blur the input image using a random-sized kernel.

    Args:
        blur_limit (int, (int, int)): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, blur_limit=7, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 3)
        self.ksize = int(random.choice(list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2))))

    def _compute_x_function(self, img, rs=None):
        return blur(img, ksize=self.ksize)

# if __name__ == '__main__':
#     from augtools.utils.test_utils import *
#     prefix = f'../test/'
#     image = prefix + 'test.jpg'
#
#     img = read_image(image)
#     # print(img)
#
#     transform = Blur()
#     result = transform(img=img, force_apply=True)
#
#     show_image(result['img'])
