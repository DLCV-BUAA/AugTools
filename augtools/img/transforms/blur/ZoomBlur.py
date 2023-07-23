
from augtools.img.transform import ImageTransform
from augtools.img.transforms.utils.img_utils import *
from scipy.ndimage import zoom as scizoom




class ZoomBlur(ImageTransform):
    """

    """
    def __init__(
        self,
        always_apply: bool = False,
        p: float = 0.5,
        severity = 1,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.severity = severity

    def _compute_x_function(self, x, rs=None):
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][self.severity - 1]

        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)

        for zoom_factor in c:
            # clipped_zoom 为什么 ? 375 * 500 -> 375 * 375 ?
            t = clipped_zoom(x, zoom_factor)
            print(t.shape)
            out += t

        x = (x + out) / (len(c) + 1)
        x = np.clip(x, 0, 1) * 255
        x = x.astype(np.uint8)
        return x


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    # print(img)

    transform = ZoomBlur()
    result = transform(img=img, force_apply=True)
    print(result['img'])

    show_image(result['img'])