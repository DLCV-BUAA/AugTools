from augtools.img.transform import ImageTransform
from augtools.img.transforms.utils.img_utils import *
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage.interpolation import map_coordinates
from PIL import Image as PILImage


class Pixelate(ImageTransform):
    def __init__(
        self,
        always_apply: bool = False,
        p: float = 0.5,
        severity = 1,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.severity = severity

    def _compute_x_function(self, x, rs=None):
        c = [0.6, 0.5, 0.4, 0.3, 0.25][self.severity - 1]

        h, w, _ = x.shape
        h_, w_ = int(h * c), int(w * c)

        x = PILImage.fromarray(x)
        x = x.resize((w_, h_), PILImage.Resampling.BOX)
        x = x.resize((w, h), PILImage.Resampling.BOX)
        x = np.array(x)

        x = x.astype(np.uint8)

        return x


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    # print(img)

    transform = Pixelate()
    result = transform(img=img, force_apply=True)
    # print(result['img'])

    show_image(result['img'])