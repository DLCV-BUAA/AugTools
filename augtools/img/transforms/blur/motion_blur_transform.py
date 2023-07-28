
from augtools.img.transform import ImageTransform
from augtools.img.transforms.utils.img_utils import *
import albumentations as A


class MotionBlur(ImageTransform):

    """
    Apply motion blur to the input image using a random-sized kernel.

    Args:
        blur_limit (Tuple): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        allow_shifted (bool): if set to true creates non shifted kernels only,
            otherwise creates randomly shifted kernels. Default: True.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image (ndarray RGB H*W*C)
    Image types:
        uint8, float32 (ndarray RGB H*W*C)
    """

    def __init__(
        self,
        blur_limit: Tuple = (3, 7), # ScaleIntType = Union[int, Tuple[int, int]]
        allow_shifted: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.allow_shifted = allow_shifted
        self.blur_limit = blur_limit
        self.motion = A.MotionBlur(blur_limit, allow_shifted, always_apply, p)

        if not allow_shifted and self.blur_limit[0] % 2 != 1 or self.blur_limit[1] % 2 != 1:
            raise ValueError(f"Blur limit must be odd when centered=True. Got: {self.blur_limit}")

    def _compute_x_function(self, img, rs=None):
        x = self.motion(image=img, force_apply=True)
        return x["image"]

        # ksize = random.choice(list(range(self.blur_limit[0], self.blur_limit[1] + 1, 2)))
        # if ksize <= 2:
        #     raise ValueError("ksize must be > 2. Got: {}".format(ksize))
        # kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        #
        # x1, x2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        # if x1 == x2:
        #     y1, y2 = random.sample(range(ksize), 2)
        # else:
        #     y1, y2 = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        #
        # # make_odd_val len_v = |v1-v2|, 如果 len_v 是奇数就返回, len_v 是偶数就把大的一边 -1 改成奇数
        # def make_odd_val(v1, v2):
        #     len_v = abs(v1 - v2) + 1
        #     if len_v % 2 != 1:
        #         if v2 > v1:
        #             v2 -= 1
        #         else:
        #             v1 -= 1
        #     return v1, v2
        #
        # if not self.allow_shifted:
        #     x1, x2 = make_odd_val(x1, x2)
        #     y1, y2 = make_odd_val(y1, y2)
        #
        #     xc = (x1 + x2) / 2
        #     yc = (y1 + y2) / 2
        #
        #     center = ksize / 2 - 0.5
        #     dx = xc - center
        #     dy = yc - center
        #     x1, x2 = [int(i - dx) for i in [x1, x2]]
        #     y1, y2 = [int(i - dy) for i in [y1, y2]]
        #
        # cv2.line(kernel, (x1, y1), (x2, y2), 1, thickness=1)
        # kernel = kernel.astype(np.float32) / np.sum(kernel)
        # FMain.convolve(img, kernel=kernel)
        # 创造一个 kernel 来 filter，什么样的 kernel 有这样的性质呢 ?


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)

    transform = MotionBlur(blur_limit=(3, 25), always_apply=True, allow_shifted=False)
    result = transform(img=img, force_apply=True)

    show_image(result['img'])