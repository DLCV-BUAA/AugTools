from augtools.img.transform import ImageTransform
from augtools.img.transforms.utils.img_utils import *
from augtools.img.functional import disk


class DefocusBlur(ImageTransform):
    def __init__(
            self,
            always_apply: bool = False,
            p: float = 0.5,
            severity: int = 1,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.severity = severity

    def _compute_x_function(self, x, rs=None):
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][self.severity - 1]

        x = np.array(x) / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        x = np.clip(channels, 0, 1) * 255
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
#     transform = DefocusBlur()
#     result = transform(img=img, force_apply=True)
#     # print(result['img'])
#
#     show_image(result['img'])
