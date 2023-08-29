from augtools.img.transform import ImageTransform
from skimage.filters import gaussian
import numpy as np


class GlassBlur(ImageTransform):
    def __init__(
            self,
            always_apply: bool = False,
            p: float = 0.5,
            severity: int = 1,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.severity = severity

    def _compute_x_function(self, x, rs=None):
        # sigma, max_delta, iterations
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][self.severity - 1]

        x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

        # locally shuffle pixels
        for i in range(c[2]):
            for h in range(224 - c[1], c[1], -1):
                for w in range(224 - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

        x = np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255
        x = x.astype(np.uint8)
        return x

# if __name__ == '__main__':
#     from augtools.utils.test_utils import *
#
#     prefix = f'../test/'
#     image = prefix + 'test.jpg'
#
#     img = read_image(image)
#     # print(img)
#
#     transform = GlassBlur()
#     result = transform(img=img, force_apply=True)
#     print(result['img'])
#
#     show_image(result['img'])
