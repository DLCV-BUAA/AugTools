from augtools.utils.decorator import *
from augtools.utils.type_conversion import *
import cv2



@process_in_chunks
@preserve_shape
def blur(img, ksize):
    # blur_fn = _maybe_process_in_chunks(cv2.blur, ksize=(ksize, ksize))
    return cv2.blur(img, ksize=(ksize, ksize))


@process_in_chunks
@preserve_shape
def gaussian_blur(img, ksize, sigma = 0):
    # When sigma=0, it is computed as `sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8`
    return cv2.GaussianBlur(img, ksize=(ksize, ksize), sigmaX=sigma)



def get_random_crop_coords(height: int, width: int, crop_height: int, crop_width: int, h_start: float, w_start: float):
    # h_start is [0, 1) and should map to [0, (height - crop_height)]  (note inclusive)
    # This is conceptually equivalent to mapping onto `range(0, (height - crop_height + 1))`
    # See: https://github.com/albumentations-team/albumentations/pull/1080
    y1 = int((height - crop_height + 1) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width + 1) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2