from augtools.img.transform import ImageTransform
from augtools.img.transforms.utils.img_utils import *
import skimage as sk
from io import BytesIO
from PIL import Image as PILImage

class JPEG(ImageTransform):
    def __init__(
        self,
        always_apply: bool = False,
        p: float = 0.5,
        severity = 1,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.severity = severity

    def _compute_x_function(self, x, rs=None):
        c = [25, 18, 15, 10, 7][self.severity - 1]
        x = PILImage.fromarray(x)

        output = BytesIO()
        x.save(output, 'JPEG', quality=c)
        x = PILImage.open(output)
        x = np.array(x)

        x = x.astype(np.uint8)
        return x


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    # print(img)

    transform = JPEG()
    result = transform(img=img, force_apply=True)
    # print(result['img'])

    show_image(result['img'])