from augtools.img.transform import ImageTransform
from augtools.img.transforms.utils.img_utils import *
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage.interpolation import map_coordinates
from PIL import Image as PILImage


class Saturate(ImageTransform):
    def __init__(
        self,
        always_apply: bool = False,
        p: float = 0.5,
        severity = 1,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.severity = severity

    def _compute_x_function(self, x, rs=None):
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][self.severity - 1]

        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = sk.color.hsv2rgb(x)

        x = np.clip(x, 0, 1) * 255

        x = x.astype(np.uint8)

        return x


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    # print(img)

    transform = Saturate()
    result = transform(img=img, force_apply=True)
    # print(result['img'])

    show_image(result['img'])