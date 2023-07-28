

from augtools.img.transform import ImageTransform
from augtools.img.transforms.utils.img_utils import *
import skimage as sk
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


class MotionBlur(ImageTransform):

    def __init__(
        self,
        always_apply: bool = False,
        p: float = 0.5,
        severity = 1,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.severity = severity

    def _compute_x_function(self, x, rs=None):

        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

        output = BytesIO()
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())

        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                         cv2.IMREAD_UNCHANGED)

        if x.shape != (224, 224):
            return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
        else:  # greyscale to RGB
            return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)



if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    # print(img)

    transform = MotionBlur()
    result = transform(img=img, force_apply=True)
    # print(result['img'])

    show_image(result['img'])