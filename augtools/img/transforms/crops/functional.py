from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from augtools.img.transform import DualTransform
from augtools.img.transforms.utils.img_utils import *
import skimage as sk
import random
from typing_extensions import Concatenate, ParamSpec
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, cast

BoxInternalType = Tuple[float, float, float, float]
BoxType = Union[BoxInternalType, Tuple[float, float, float, float, Any]]
TBox = TypeVar("TBox", BoxType, BoxInternalType)
NumType = Union[int, float, np.ndarray]
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