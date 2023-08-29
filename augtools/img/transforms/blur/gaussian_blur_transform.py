import warnings
import random
from augtools.img.transform import ImageTransform
from augtools.img.transforms.utils.img_utils import *


class GaussianBlur(ImageTransform):
    """Blur the input image using a Gaussian filter with a random kernel size.

    Args:
        blur_limit (int, (int, int)): maximum Gaussian kernel size for blurring the input image.
            Must be zero or odd and in range [0, inf). If set to 0 it will be computed from sigma
            as `round(sigma * (3 if img.dtype == np.uint8 else 4) * 2 + 1) + 1`.
            If set single value `blur_limit` will be in range (0, blur_limit).
            Default: (3, 7).
        sigma_limit (float, (float, float)): Gaussian kernel standard deviation. Must be in range [0, inf).
            If set single value `sigma_limit` will be in range (0, sigma_limit).
            If set to 0 sigma will be computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`. Default: 0.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(
            self,
            blur_limit=(3, 7),
            sigma_limit=0,
            always_apply=False,
            p=0.5,
    ):
        super().__init__(always_apply, p)
        self.blur_limit = to_tuple(blur_limit, 0)
        self.sigma_limit = to_tuple(sigma_limit if sigma_limit is not None else 0, 0)

        if self.blur_limit[0] == 0 and self.sigma_limit[0] == 0:
            self.blur_limit = 3, max(3, self.blur_limit[1])
            warnings.warn(
                "blur_limit and sigma_limit minimum value can not be both equal to 0. "
                "blur_limit minimum value changed to 3."
            )

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
                self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("GaussianBlur supports only odd blur limits.")

        self.ksize = random.randrange(self.blur_limit[0], self.blur_limit[1] + 1)
        if self.ksize != 0 and self.ksize % 2 != 1:
            self.ksize = (self.ksize + 1) % (self.blur_limit[1] + 1)
        self.sigma = random.uniform(*self.sigma_limit)

    def _compute_x_function(self, img, rs=None):
        return gaussian_blur(img, ksize=self.ksize, sigma=self.sigma)

# if __name__ == '__main__':
#     from augtools.utils.test_utils import *
#     prefix = f'../test/'
#     image = prefix + 'test.jpg'
#
#     img = read_image(image)
#
#     transform = GaussianBlur()
#     result = transform(img=img, force_apply=True)
#     print(result['img'])
#
#     show_image(result['img'])
