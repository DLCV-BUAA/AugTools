from augtools.img.transforms.utils.img_utils import *
from typing import Any, Optional, Sequence, Tuple, TypeVar, cast
import math
from augtools.img import random_utils
from scipy.ndimage import gaussian_filter


NumType = Union[int, float, np.ndarray]
BoxInternalType = Tuple[float, float, float, float]
BoxType = Union[BoxInternalType, Tuple[float, float, float, float, Any]]
TBox = TypeVar("TBox", BoxType, BoxInternalType)
BoxInternalType = Tuple[float, float, float, float]
BoxType = Union[BoxInternalType, Tuple[float, float, float, float, Any]]
KeypointInternalType = Tuple[float, float, float, float]
KeypointType = Union[KeypointInternalType, Tuple[float, float, float, float, Any]]
ImageColorType = Union[float, Sequence[float]]
ScaleFloatType = Union[float, Tuple[float, float]]
ScaleIntType = Union[int, Tuple[int, int]]
FillValueType = Optional[Union[int, float, Sequence[int], Sequence[float]]]


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
                        chunk = img[:, :, index + i: index + i + 1]
                        chunk = process_fn(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index: index + 4]
                    chunk = process_fn(chunk, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = process_fn(img, **kwargs)
        return img

    return __process_fn


def get_center_crop_coords(height: int, width: int, crop_height: int, crop_width: int):
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    return x1, y1, x2, y2


def center_crop(img: np.ndarray, crop_height: int, crop_width: int):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_center_crop_coords(height, width, crop_height, crop_width)
    img = img[y1:y2, x1:x2]
    return img


def normalize_bbox(bbox: TBox, rows: int, cols: int) -> TBox:
    """Normalize coordinates of a bounding box. Divide x-coordinates by image width and y-coordinates
    by image height.

    Args:
        bbox: Denormalized bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image height.
        cols: Image width.

    Returns:
        Normalized bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If rows or cols is less or equal zero

    """

    if rows <= 0:
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")

    tail: Tuple[Any, ...]
    (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])

    x_min, x_max = x_min / cols, x_max / cols
    y_min, y_max = y_min / rows, y_max / rows

    return cast(BoxType, (x_min, y_min, x_max, y_max) + tail)  # type: ignore


def denormalize_bbox(bbox: TBox, rows: int, cols: int) -> TBox:
    """Denormalize coordinates of a bounding box. Multiply x-coordinates by image width and y-coordinates
    by image height. This is an inverse operation for :func:`~albumentations.augmentations.bbox.normalize_bbox`.

    Args:
        bbox: Normalized bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image height.
        cols: Image width.

    Returns:
        Denormalized bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If rows or cols is less or equal zero

    """
    tail: Tuple[Any, ...]
    (x_min, y_min, x_max, y_max), tail = bbox[:4], tuple(bbox[4:])

    if rows <= 0:
        raise ValueError("Argument rows must be positive integer")
    if cols <= 0:
        raise ValueError("Argument cols must be positive integer")

    x_min, x_max = x_min * cols, x_max * cols
    y_min, y_max = y_min * rows, y_max * rows

    return cast(BoxType, (x_min, y_min, x_max, y_max) + tail)  # type: ignore


def crop_bbox_by_coords(
        bbox: BoxInternalType,
        crop_coords: Tuple[int, int, int, int],
        crop_height: int,
        crop_width: int,
        rows: int,
        cols: int,
):
    """Crop a bounding box using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.

    Args:
        bbox (tuple): A cropped box `(x_min, y_min, x_max, y_max)`.
        crop_coords (tuple): Crop coordinates `(x1, y1, x2, y2)`.
        crop_height (int):
        crop_width (int):
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        tuple: A cropped bounding box `(x_min, y_min, x_max, y_max)`.

    """
    bbox = denormalize_bbox(bbox, rows, cols)
    x_min, y_min, x_max, y_max = bbox[:4]
    x1, y1, _, _ = crop_coords
    cropped_bbox = x_min - x1, y_min - y1, x_max - x1, y_max - y1
    return normalize_bbox(cropped_bbox, crop_height, crop_width)


def bbox_center_crop(bbox: BoxInternalType, crop_height: int, crop_width: int, rows: int, cols: int):
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def crop_keypoint_by_coords(
        keypoint: KeypointInternalType, crop_coords: Tuple[int, int, int, int]
):  # skipcq: PYL-W0613
    """Crop a keypoint using the provided coordinates of bottom-left and top-right corners in pixels and the
    required height and width of the crop.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        crop_coords (tuple): Crop box coords `(x1, x2, y1, y2)`.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    x1, y1, _, _ = crop_coords
    return x - x1, y - y1, angle, scale


def keypoint_center_crop(keypoint: KeypointInternalType, crop_height: int, crop_width: int, rows: int, cols: int):
    """Keypoint center crop.

    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        crop_height (int): Crop height.
        crop_width (int): Crop width.
        rows (int): Image height.
        cols (int): Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    crop_coords = get_center_crop_coords(rows, cols, crop_height, crop_width)
    return crop_keypoint_by_coords(keypoint, crop_coords)


def crop(img: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int):
    height, width = img.shape[:2]
    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
            )
        )

    if x_min < 0 or x_max > width or y_min < 0 or y_max > height:
        raise ValueError(
            "Values for crop should be non negative and equal or smaller than image sizes"
            "(x_min = {x_min}, y_min = {y_min}, x_max = {x_max}, y_max = {y_max}, "
            "height = {height}, width = {width})".format(
                x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, height=height, width=width
            )
        )

    return img[y_min:y_max, x_min:x_max]


def bbox_crop(bbox: BoxInternalType, x_min: int, y_min: int, x_max: int, y_max: int, rows: int, cols: int):
    """Crop a bounding box.

    Args:
        bbox (tuple): A bounding box `(x_min, y_min, x_max, y_max)`.
        x_min (int):
        y_min (int):
        x_max (int):
        y_max (int):
        rows (int): Image rows.
        cols (int): Image cols.

    Returns:
        tuple: A cropped bounding box `(x_min, y_min, x_max, y_max)`.

    """
    crop_coords = x_min, y_min, x_max, y_max
    crop_height = y_max - y_min
    crop_width = x_max - x_min
    return crop_bbox_by_coords(bbox, crop_coords, crop_height, crop_width, rows, cols)


def vflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ...])


def hflip(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[:, ::-1, ...])


def hflip_cv2(img: np.ndarray) -> np.ndarray:
    return cv2.flip(img, 1)


def random_flip(img: np.ndarray, code: int) -> np.ndarray:
    return cv2.flip(img, code)


# Tuple[float, float, float, float]
def bbox_vflip(bbox, rows, cols):
    """Flip a bounding box vertically around the x-axis.

    Args:
        bbox(Tuple[float, float, float, float]): A bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image rows.
        cols: Image cols.

    Returns:
        tuple: A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    y_min_new = rows - y_max
    y_max_new = rows - y_min
    return (x_min, y_min_new, x_max, y_max_new)


def bbox_hflip(bbox, rows, cols):
    """Flip a bounding box horizontally around the y-axis.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    x_min_new = cols - x_max
    x_max_new = cols - x_min
    return x_min_new, y_min, x_max_new, y_max


def bbox_flip(bbox, d: int, rows: int, cols: int):
    """Flip a bounding box either vertically, horizontally or both depending on the value of `d`.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        d: dimension. 0 for vertical flip, 1 for horizontal, -1 for transpose
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        bbox = bbox_vflip(bbox, rows, cols)
    elif d == 1:
        bbox = bbox_hflip(bbox, rows, cols)
    elif d == -1:
        bbox = bbox_hflip(bbox, rows, cols)
        bbox = bbox_vflip(bbox, rows, cols)
    else:
        raise ValueError("Invalid d value {}. Valid values are -1, 0 and 1".format(d))
    return bbox


# Tuple[float, float, float, float]
def keypoint_vflip(keypoint, rows: int, cols: int):
    """Flip a keypoint vertically around the x-axis.

    Args:
        keypoint(Tuple[float, float, float, float]): A keypoint `(x, y, angle, scale)`.
        rows: Image height.
        cols: Image width.

    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    angle = -angle
    return (x, (rows - 1) - y, angle, scale)


def keypoint_hflip(keypoint: KeypointInternalType, rows: int, cols: int) -> KeypointInternalType:
    """Flip a keypoint horizontally around the y-axis.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        rows: Image height.
        cols: Image width.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    angle = math.pi - angle
    return ((cols - 1) - x, y, angle, scale)


def keypoint_flip(keypoint: KeypointInternalType, d: int, rows: int, cols: int) -> KeypointInternalType:
    """Flip a keypoint either vertically, horizontally or both depending on the value of `d`.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        d: Number of flip. Must be -1, 0 or 1:
            * 0 - vertical flip,
            * 1 - horizontal flip,
            * -1 - vertical and horizontal flip.
        rows: Image height.
        cols: Image width.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    Raises:
        ValueError: if value of `d` is not -1, 0 or 1.

    """
    if d == 0:
        keypoint = keypoint_vflip(keypoint, rows, cols)
    elif d == 1:
        keypoint = keypoint_hflip(keypoint, rows, cols)
    elif d == -1:
        keypoint = keypoint_hflip(keypoint, rows, cols)
        keypoint = keypoint_vflip(keypoint, rows, cols)
    else:
        raise ValueError(f"Invalid d value {d}. Valid values are -1, 0 and 1")
    return keypoint


# ----------------------------------------------------------------------------------

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


def transpose(img: np.ndarray) -> np.ndarray:
    return img.transpose(1, 0, 2) if len(img.shape) > 2 else img.transpose(1, 0)

def bbox_transpose(
    bbox: KeypointInternalType, axis: int
) -> KeypointInternalType:  # skipcq: PYL-W0613
    """Transposes a bounding box along given axis.

    Args:
        bbox: A bounding box `(x_min, y_min, x_max, y_max)`.
        axis: 0 - main axis, 1 - secondary axis.
        rows: Image rows.
        cols: Image cols.

    Returns:
        A bounding box tuple `(x_min, y_min, x_max, y_max)`.

    Raises:
        ValueError: If axis not equal to 0 or 1.

    """
    x_min, y_min, x_max, y_max = bbox[:4]
    if axis not in {0, 1}:
        raise ValueError("Axis must be either 0 or 1.")
    if axis == 0:
        bbox = (y_min, x_min, y_max, x_max)
    if axis == 1:
        bbox = (1 - y_max, 1 - x_max, 1 - y_min, 1 - x_min)
    return bbox


def keypoint_transpose(keypoint: KeypointInternalType) -> KeypointInternalType:
    """Rotate a keypoint by angle.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]

    if angle <= np.pi:
        angle = np.pi - angle
    else:
        angle = 3 * np.pi - angle

    return y, x, angle, scale



def bbox_from_mask(mask):
    """Create bounding box from binary mask (fast version)

    Args:
        mask (numpy.ndarray): binary mask.

    Returns:
        tuple: A bounding box tuple `(x_min, y_min, x_max, y_max)`.

    """
    rows = np.any(mask, axis=1)
    if not rows.any():
        return -1, -1, -1, -1
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return x_min, y_min, x_max + 1, y_max + 1



def elastic_transform(
    img: np.ndarray,
    alpha: float,
    sigma: float,
    alpha_affine: float,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_REFLECT_101,
    value: Optional[ImageColorType] = None,
    random_state: Optional[np.random.RandomState] = None,
    approximate: bool = False,
    same_dxdy: bool = False,
):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    height, width = img.shape[:2]

    # Random affine
    center_square = np.array((height, width), dtype=np.float32) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.array(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ],
        dtype=np.float32,
    )
    pts2 = pts1 + random_utils.uniform(-alpha_affine, alpha_affine, size=pts1.shape, random_state=random_state).astype(
        np.float32
    )
    matrix = cv2.getAffineTransform(pts1, pts2)

    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value
    )
    img = warp_fn(img)

    if approximate:
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        dx = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
        dx *= alpha
        if same_dxdy:
            # Speed up even more
            dy = dx
        else:
            dy = random_utils.rand(height, width, random_state=random_state).astype(np.float32) * 2 - 1
            cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
            dy *= alpha
    else:
        dx = np.float32(
            gaussian_filter((random_utils.rand(height, width, random_state=random_state) * 2 - 1), sigma) * alpha
        )
        if same_dxdy:
            # Speed up
            dy = dx
        else:
            dy = np.float32(
                gaussian_filter((random_utils.rand(height, width, random_state=random_state) * 2 - 1), sigma) * alpha
            )

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    remap_fn = _maybe_process_in_chunks(
        cv2.remap, map1=map_x, map2=map_y, interpolation=interpolation, borderMode=border_mode, borderValue=value
    )
    return remap_fn(img)


def resize(img, height, width, interpolation=cv2.INTER_LINEAR):
    img_height, img_width = img.shape[:2]
    if height == img_height and width == img_width:
        return img
    resize_fn = _maybe_process_in_chunks(cv2.resize, dsize=(width, height), interpolation=interpolation)
    return resize_fn(img)


def perspective(
    img: np.ndarray,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    border_val: Union[int, float, List[int], List[float], np.ndarray],
    border_mode: int,
    keep_size: bool,
    interpolation: int,
):
    h, w = img.shape[:2]
    perspective_func = _maybe_process_in_chunks(
        cv2.warpPerspective,
        M=matrix,
        dsize=(max_width, max_height),
        borderMode=border_mode,
        borderValue=border_val,
        flags=interpolation,
    )
    warped = perspective_func(img)

    if keep_size:
        return resize(warped, h, w, interpolation=interpolation)

    return warped


def perspective_bbox(
    bbox: BoxInternalType,
    height: int,
    width: int,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
) -> BoxInternalType:
    x1, y1, x2, y2 = denormalize_bbox(bbox, height, width)[:4]

    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

    x1, y1, x2, y2 = float("inf"), float("inf"), 0, 0
    for pt in points:
        pt = perspective_keypoint(pt.tolist() + [0, 0], height, width, matrix, max_width, max_height, keep_size)
        x, y = pt[:2]
        x1 = min(x1, x)
        x2 = max(x2, x)
        y1 = min(y1, y)
        y2 = max(y2, y)

    return normalize_bbox((x1, y1, x2, y2), height if keep_size else max_height, width if keep_size else max_width)


def rotation2DMatrixToEulerAngles(matrix: np.ndarray, y_up: bool = False) -> float:
    """
    Args:
        matrix (np.ndarray): Rotation matrix
        y_up (bool): is Y axis looks up or down
    """
    if y_up:
        return np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.arctan2(-matrix[1, 0], matrix[0, 0])


KeypointInternalType = Tuple[float, float, float, float]

def keypoint_scale(keypoint: KeypointInternalType, scale_x: float, scale_y: float) -> KeypointInternalType:
    """Scales a keypoint by scale_x and scale_y.

    Args:
        keypoint: A keypoint `(x, y, angle, scale)`.
        scale_x: Scale coefficient x-axis.
        scale_y: Scale coefficient y-axis.

    Returns:
        A keypoint `(x, y, angle, scale)`.

    """
    x, y, angle, scale = keypoint[:4]
    return x * scale_x, y * scale_y, angle, scale * max(scale_x, scale_y)


def perspective_keypoint(
    keypoint: KeypointInternalType,
    height: int,
    width: int,
    matrix: np.ndarray,
    max_width: int,
    max_height: int,
    keep_size: bool,
) -> KeypointInternalType:
    x, y, angle, scale = keypoint

    keypoint_vector = np.array([x, y], dtype=np.float32).reshape([1, 1, 2])

    x, y = cv2.perspectiveTransform(keypoint_vector, matrix)[0, 0]
    angle += rotation2DMatrixToEulerAngles(matrix[:2, :2], y_up=True)

    scale_x = np.sign(matrix[0, 0]) * np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
    scale_y = np.sign(matrix[1, 1]) * np.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
    scale *= max(scale_x, scale_y)

    if keep_size:
        scale_x = width / max_width
        scale_y = height / max_height
        return keypoint_scale((x, y, angle, scale), scale_x, scale_y)

    return x, y, angle, scale

