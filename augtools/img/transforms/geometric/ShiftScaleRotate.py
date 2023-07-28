from augtools.img.transform import DualTransform
from augtools.img.transforms.utils.img_utils import *
import skimage as sk
import random
from typing_extensions import Concatenate, ParamSpec
import math


def get_num_channels(image: np.ndarray) -> int:
    return image.shape[2] if len(image.shape) == 3 else 1


def _maybe_process_in_chunks(
    process_fn, **kwargs
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap OpenCV function to enable processing images with more than 4 channels.

    Limitations:
        This wrapper requires image to be the first argument and rest must be sent via named arguments.

    Args:
        process_fn: Transform function (e.g cv2.resize).
        kwargs: Additional parameters.

    Returns:
        numpy.ndarray: Transformed image.

    """

    @wraps(process_fn)
    def __process_fn(img: np.ndarray) -> np.ndarray:
        num_channels = get_num_channels(img)
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    # Many OpenCV functions cannot work with 2-channel images
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def shift_scale_rotate(
    img, angle, scale, dx, dy, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT_101, value=None
):
    height, width = img.shape[:2]
    # for images we use additional shifts of (0.5, 0.5) as otherwise
    # we get an ugly black border for 90deg rotations
    center = (width / 2 - 0.5, height / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    warp_affine_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    return warp_affine_fn(img)



def bbox_rotate(bbox: Tuple, angle: float, method: str, rows: int, cols: int):
    """Rotates a bounding box by angle degrees.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        angle: Angle of rotation in degrees.
        method: Rotation method used. Should be one of: "largest_box", "ellipse". Default: "largest_box".
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    References:
        https://arxiv.org/abs/2109.13488

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    scale = cols / float(rows)
    if method == "largest_box":
        x = np.array([x_min, x_max, x_max, x_min]) - 0.5
        y = np.array([y_min, y_min, y_max, y_max]) - 0.5
    elif method == "ellipse":
        w = (x_max - x_min) / 2
        h = (y_max - y_min) / 2
        data = np.arange(0, 360, dtype=np.float32)
        x = w * np.sin(np.radians(data)) + (w + x_min - 0.5)
        y = h * np.cos(np.radians(data)) + (h + y_min - 0.5)
    else:
        raise ValueError(f"Method {method} is not a valid rotation method.")
    angle = np.deg2rad(angle)
    x_t = (np.cos(angle) * x * scale + np.sin(angle) * y) / scale
    y_t = -np.sin(angle) * x * scale + np.cos(angle) * y
    x_t = x_t + 0.5
    y_t = y_t + 0.5

    x_min, x_max = min(x_t), max(x_t)
    y_min, y_max = min(y_t), max(y_t)

    return x_min, y_min, x_max, y_max



def bbox_shift_scale_rotate(bbox, angle, scale, dx, dy, rotate_method, rows, cols):  # skipcq: PYL-W0613
    """Rotates, shifts and scales a bounding box. Rotation is made by angle degrees,
    scaling is made by scale factor and shifting is made by dx and dy.


    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        angle (int): Angle of rotation in degrees.
        scale (int): Scale factor.
        dx (int): Shift along x-axis in pixel units.
        dy (int): Shift along y-axis in pixel units.
        rotate_method(str): Rotation method used. Should be one of: "largest_box", "ellipse".
            Default: "largest_box".
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    height, width = rows, cols
    center = (width / 2, height / 2)
    if rotate_method == "ellipse":
        x_min, y_min, x_max, y_max = bbox_rotate(bbox, angle, rotate_method, rows, cols)
        matrix = cv2.getRotationMatrix2D(center, 0, scale)
    else:
        x_min, y_min, x_max, y_max = bbox[:4]
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height
    x = np.array([x_min, x_max, x_max, x_min])
    y = np.array([y_min, y_min, y_max, y_max])
    ones = np.ones(shape=(len(x)))
    points_ones = np.vstack([x, y, ones]).transpose()
    points_ones[:, 0] *= width
    points_ones[:, 1] *= height
    tr_points = matrix.dot(points_ones.T).T
    tr_points[:, 0] /= width
    tr_points[:, 1] /= height

    x_min, x_max = min(tr_points[:, 0]), max(tr_points[:, 0])
    y_min, y_max = min(tr_points[:, 1]), max(tr_points[:, 1])

    return x_min, y_min, x_max, y_max


def keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, rows, cols):
    (
        x,
        y,
        a,
        s,
    ) = keypoint[:4]
    height, width = rows, cols
    center = (cols - 1) * 0.5, (rows - 1) * 0.5
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx * width
    matrix[1, 2] += dy * height

    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()
    angle = a + math.radians(angle)
    scale = s * scale

    return x, y, angle, scale



class ShiftScaleRotate(DualTransform):
    """Randomly apply affine transforms: translate, scale and rotate the input.
       随机平移、旋转，缩放

    Args:
        shift_limit ((float, float) or float): shift factor range for both height and width. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
        shift_limit_x ((float, float) or float): shift factor range for width. If it is set then this value
            instead of shift_limit will be used for shifting width.  If shift_limit_x is a single float value,
            the range will be (-shift_limit_x, shift_limit_x). Absolute values for lower and upper bounds should lie in
            the range [0, 1]. Default: None.
        shift_limit_y ((float, float) or float): shift factor range for height. If it is set then this value
            instead of shift_limit will be used for shifting height.  If shift_limit_y is a single float value,
            the range will be (-shift_limit_y, shift_limit_y). Absolute values for lower and upper bounds should lie
            in the range [0, 1]. Default: None.

        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).

        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: (-45, 45).

        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of int,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        rotate_method (str): rotation method used for the bounding boxes. Should be one of "largest_box" or "ellipse".
            Default: "largest_box"

        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        shift_limit=0.0625,
        shift_limit_x=None,
        shift_limit_y=None,

        scale_limit=0.1,
        rotate_limit=45,

        # 插值用的一些内容 ?
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        rotate_method="largest_box",

        always_apply=False,
        p=0.5,
    ):
        super(ShiftScaleRotate, self).__init__(always_apply, p)
        self.shift_limit_x = to_tuple(shift_limit_x if shift_limit_x is not None else shift_limit)
        self.shift_limit_y = to_tuple(shift_limit_y if shift_limit_y is not None else shift_limit)

        self.scale_limit = to_tuple(scale_limit, bias=1.0)

        self.rotate_limit = to_tuple(rotate_limit)

        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

        self.rotate_method = rotate_method
        if self.rotate_method not in ["largest_box", "ellipse"]:
            raise ValueError(f"Rotation method {self.rotate_method} is not valid.")

    def __call__(self, *args, force_apply: bool = False, **kwargs):
        if (random.random() < self.p) and not self.always_apply and not force_apply:
            return kwargs

        res = {}
        angle = random.uniform(self.rotate_limit[0], self.rotate_limit[1])
        scale = random.uniform(self.scale_limit[0], self.scale_limit[1])
        dx = random.uniform(self.shift_limit_x[0], self.shift_limit_x[1])
        dy = random.uniform(self.shift_limit_y[0], self.shift_limit_y[1])
        height, width = img.shape[0], img.shape[1]

        if "img" in kwargs:
            res["img"] = shift_scale_rotate(img, angle, scale, dx, dy, self.interpolation, self.border_mode, self.value)

        if "bbox" in kwargs:
            res["bbox"] = bbox_shift_scale_rotate(kwargs["bbox"], angle, scale, dx, dy, self.rotate_method, height, width)

        if "keypoint" in kwargs:
            res["keypoint"] = keypoint_shift_scale_rotate(kwargs["keypoint"], angle, scale, dx, dy, height, width)

        return res


    # def apply(self, img, angle=0, scale=0, dx=0, dy=0, interpolation=cv2.INTER_LINEAR):
    #     return F.shift_scale_rotate(img, angle, scale, dx, dy, interpolation, self.border_mode, self.value)
    #
    # def apply_to_mask(self, img, angle=0, scale=0, dx=0, dy=0):
    #     return F.shift_scale_rotate(img, angle, scale, dx, dy, cv2.INTER_NEAREST, self.border_mode, self.mask_value)
    #
    # def apply_to_keypoint(self, keypoint, angle=0, scale=0, dx=0, dy=0, rows=0, cols=0):
    #     return F.keypoint_shift_scale_rotate(keypoint, angle, scale, dx, dy, rows, cols)
    #
    # def apply_to_bbox(self, bbox, angle, scale, dx, dy, **params):
    #     return F.bbox_shift_scale_rotate(bbox, angle, scale, dx, dy, self.rotate_method, **params)


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    # print(img)
    bbox = (50, 60, 50, 80)

    transform = ShiftScaleRotate(shift_limit=0, rotate_limit=0, scale_limit=0)
    result = transform(img=img, force_apply=True, bbox=bbox)
    # print(result['img'])
    print(result["bbox"])
    show_image(result['img'])