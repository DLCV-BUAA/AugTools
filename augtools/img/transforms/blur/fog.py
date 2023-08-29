from augtools.img.transform import ImageTransform
import numpy as np
from augtools.img.functional import plasma_fractal


class FogBlur(ImageTransform):
    def __init__(
            self,
            always_apply: bool = False,
            p: float = 0.5,
            severity: int = 1,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.severity = severity

    def _compute_x_function(self, x, rs=None):
        c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][self.severity - 1]

        x = np.array(x) / 255.
        max_val = x.max()

        h, w, _ = x.shape
        map_size = max(h, w)

        p, j = 0, 1
        while j < map_size:
            p, j = p + 1, j * 2
        map_size = j

        x += c[0] * plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:h, :w][..., np.newaxis]
        x = np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
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
#     transform = FogBlur()
#     result = transform(img=img, force_apply=True)
#     # print(result['img'])
#
#     show_image(result['img'])
