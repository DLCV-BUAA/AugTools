from augtools.img.transform import ImageTransform
import numpy as np
import cv2


class FrostBlur(ImageTransform):
    def __init__(
            self,
            always_apply: bool = False,
            p: float = 0.5,
            severity: int = 1,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.severity = severity

    def _compute_x_function(self, x, rs=None):
        c = [(1, 0.4),
             (0.8, 0.6),
             (0.7, 0.7),
             (0.65, 0.7),
             (0.6, 0.75)][self.severity - 1]

        idx = np.random.randint(5)
        filename = ['./FrostResource/frost1.png', './FrostResource/frost2.png', './FrostResource/frost3.png',
                    './FrostResource/frost4.jpg', './FrostResource/frost5.jpg', './FrostResource/frost6.jpg'][idx]
        frost = cv2.imread(filename)

        # randomly crop and convert to rgb
        x_start, y_start = np.random.randint(0, frost.shape[0] - 224), np.random.randint(0, frost.shape[1] - 224)
        frost = frost[x_start:x_start + 224, y_start:y_start + 224][..., [2, 1, 0]]

        h, w, _ = x.shape
        t = cv2.resize(frost, (w, h))

        x = np.clip(c[0] * np.array(x) + c[1] * t, 0, 255)
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
#     transform = FrostBlur()
#     result = transform(img=img, force_apply=True)
#     # print(result['img'])
#
#     show_image(result['img'])
