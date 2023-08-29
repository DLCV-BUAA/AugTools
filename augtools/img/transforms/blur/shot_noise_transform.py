from augtools.img.transform import ImageTransform
from augtools.img.transforms.utils.img_utils import *


class ShotBlur(ImageTransform):
    def __init__(
            self,
            always_apply: bool = False,
            p: float = 0.5,
            severity: int = 1,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.severity = severity

    def _compute_x_function(self, x, rs=None):
        c = [60, 25, 12, 5, 3][self.severity - 1]

        x = x / 255.
        x = np.clip(np.random.poisson(x * c) / c, 0, 1) * 255
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
#     transform = ShotBlur()
#     result = transform(img=img, force_apply=True)
#     print(result['img'])
#
#     show_image(result['img'])
