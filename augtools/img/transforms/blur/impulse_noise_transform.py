from augtools.img.transform import ImageTransform
import skimage as sk
import numpy as np


class ImpulseBlur(ImageTransform):
    def __init__(
            self,
            always_apply: bool = False,
            p: float = 0.5,
            severity: int = 1,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.severity = severity

    def _compute_x_function(self, x, rs=None):
        c = [.03, .06, .09, 0.17, 0.27][self.severity - 1]

        # impulse 脉冲噪声，amount 是概率遭到脉冲，以及怎么被遭到脉冲，这里 mode = s&p
        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        x = np.clip(x, 0, 1) * 255
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
#     transform = ImpulseBlur()
#     result = transform(img=img, force_apply=True)
#     # print(result['img'])
#
#     show_image(result['img'])
