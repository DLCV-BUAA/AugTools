
import random
from augtools.img.transform import ImageTransform
from augtools.img.transforms.utils.img_utils import *


def channel_dropout(
    img: np.ndarray, channels_to_drop: Union[int, Tuple[int, ...], np.ndarray], fill_value: Union[int, float] = 0
) -> np.ndarray:
    if len(img.shape) == 2 or img.shape[2] == 1:
        raise NotImplementedError("Only one channel. ChannelDropout is not defined.")

    img = img.copy()
    img[..., channels_to_drop] = fill_value
    return img


class ChannelDropout(ImageTransform):
    """Randomly Drop Channels in the input Image.

    Args:
        channel_drop_range (int, int): range from which we choose the number of channels to drop.
        fill_value (int, float): pixel value for the dropped channel.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, unit32, float32
    """

    def __init__(
        self,
        channel_drop_range: Tuple[int, int] = (1, 1),
        fill_value: Union[int, float] = 0,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(ChannelDropout, self).__init__(always_apply, p)

        self.channel_drop_range = channel_drop_range

        self.min_channels = channel_drop_range[0]
        self.max_channels = channel_drop_range[1]

        if not 1 <= self.min_channels <= self.max_channels:
            raise ValueError("Invalid channel_drop_range. Got: {}".format(channel_drop_range))

        self.fill_value = fill_value

    def _compute_x_function(self, img, rs=None):
        num_channels = img.shape[-1]

        if len(img.shape) == 2 or num_channels == 1:
            raise NotImplementedError("Images has one channel. ChannelDropout is not defined.")

        if self.max_channels >= num_channels:
            raise ValueError("Can not drop all channels in ChannelDropout.")

        num_drop_channels = random.randint(self.min_channels, self.max_channels)

        channels_to_drop = random.sample(range(num_channels), k=num_drop_channels)

        return channel_dropout(img, channels_to_drop, self.fill_value)


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)

    transform = ChannelDropout()
    result = transform(img=img, force_apply=True)
    print(result['img'])

    show_image(result['img'])
