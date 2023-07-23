from augtools.img.transform import DualTransform
from augtools.img.transforms.utils.img_utils import *
import skimage as sk
import random
from typing_extensions import Concatenate, ParamSpec
import math
import random_utils
from scipy.ndimage import gaussian_filter

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, cast
ImageColorType = Union[float, Sequence[float]]
BoxInternalType = Tuple[float, float, float, float]
BoxType = Union[BoxInternalType, Tuple[float, float, float, float, Any]]
TBox = TypeVar("TBox", BoxType, BoxInternalType)

# --------------------
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


class Perspective(DualTransform):
    """Perform a random four point perspective transform of the input.

    Args:
        scale (float or (float, float)): standard deviation of the normal distributions. These are used to sample
            the random distances of the subimage's corners from the full image's corners.
            If scale is a single float value, the range will be (0, scale). Default: (0.05, 0.1).
        keep_size (bool): Whether to resize image’s back to their original size after applying the perspective
            transform. If set to False, the resulting images may end up having different shapes
            and will always be a list, never an array. Default: True
        pad_mode (OpenCV flag): OpenCV border mode.
        pad_val (int, float, list of int, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
            Default: 0
        mask_pad_val (int, float, list of int, list of float): padding value for mask
            if border_mode is cv2.BORDER_CONSTANT. Default: 0
        fit_output (bool): If True, the image plane size and position will be adjusted to still capture
            the whole image after perspective transformation. (Followed by image resizing if keep_size is set to True.)
            Otherwise, parts of the transformed image may be outside of the image plane.
            This setting should not be set to True when using large scale values as it could lead to very large images.
            Default: False
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        scale=(0.05, 0.1),
        keep_size=True,
        pad_mode=cv2.BORDER_CONSTANT,
        pad_val=0,
        mask_pad_val=0,
        fit_output=False,
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.scale = to_tuple(scale, 0)
        self.keep_size = keep_size
        self.pad_mode = pad_mode
        self.pad_val = pad_val
        self.mask_pad_val = mask_pad_val
        self.fit_output = fit_output
        self.interpolation = interpolation

    def __call__(self, *args, force_apply: bool = False, **kwargs):
        if (random.random() < self.p) and not self.always_apply and not force_apply:
            return kwargs
        res = {}

        params = {"image": kwargs["img"]}
        params_ = self.get_params_dependent_on_targets(params=params)

        res["img"] = perspective(
            kwargs["img"],
            params_["matrix"],
            params_["max_width"],
            params_["max_height"],
            self.pad_val,
            self.pad_mode,
            self.keep_size,
            params_["interpolation"]
        )

        if "bbox" in kwargs:
            res["bbox"] = perspective_bbox(kwargs["bbox"],
                                           kwargs["img"].shape[0],
                                           kwargs["img"].shape[1],
                                           params_["matrix"],
                                           params_["max_width"],
                                           params_["max_height"],
                                           self.keep_size
                                           )

        return res

    # def apply(self, img, matrix=None, max_height=None, max_width=None, **params):
    #     return F.perspective(
    #         img, matrix, max_width, max_height, self.pad_val, self.pad_mode, self.keep_size, params["interpolation"]
    #     )
    #
    # def apply_to_bbox(self, bbox, matrix=None, max_height=None, max_width=None, **params):
    #     return F.perspective_bbox(bbox, params["rows"], params["cols"], matrix, max_width, max_height, self.keep_size)
    #
    # def apply_to_keypoint(self, keypoint, matrix=None, max_height=None, max_width=None, **params):
    #     return F.perspective_keypoint(
    #         keypoint, params["rows"], params["cols"], matrix, max_width, max_height, self.keep_size
    #     )

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params):
        h, w = params["image"].shape[:2]

        scale = random_utils.uniform(*self.scale)
        points = random_utils.normal(0, scale, [4, 2])
        points = np.mod(np.abs(points), 1)

        # top left -- no changes needed, just use jitter
        # top right
        points[1, 0] = 1.0 - points[1, 0]  # w = 1.0 - jitter
        # bottom right
        points[2] = 1.0 - points[2]  # w = 1.0 - jitt
        # bottom left
        points[3, 1] = 1.0 - points[3, 1]  # h = 1.0 - jitter

        points[:, 0] *= w
        points[:, 1] *= h

        # Obtain a consistent order of the points and unpack them individually.
        # Warning: don't just do (tl, tr, br, bl) = _order_points(...)
        # here, because the reordered points is used further below.
        points = self._order_points(points)
        tl, tr, br, bl = points

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        min_width = None
        max_width = None
        while min_width is None or min_width < 2:
            width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            max_width = int(max(width_top, width_bottom))
            min_width = int(min(width_top, width_bottom))
            if min_width < 2:
                step_size = (2 - min_width) / 2
                tl[0] -= step_size
                tr[0] += step_size
                bl[0] -= step_size
                br[0] += step_size

        # compute the height of the new image, which will be the maximum distance between the top-right
        # and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
        min_height = None
        max_height = None
        while min_height is None or min_height < 2:
            height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = int(max(height_right, height_left))
            min_height = int(min(height_right, height_left))
            if min_height < 2:
                step_size = (2 - min_height) / 2
                tl[1] -= step_size
                tr[1] -= step_size
                bl[1] += step_size
                br[1] += step_size

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left order
        # do not use width-1 or height-1 here, as for e.g. width=3, height=2
        # the bottom right coordinate is at (3.0, 2.0) and not (2.0, 1.0)
        dst = np.array([[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]], dtype=np.float32)

        # compute the perspective transform matrix and then apply it
        m = cv2.getPerspectiveTransform(points, dst)

        if self.fit_output:
            m, max_width, max_height = self._expand_transform(m, (h, w))

        return {"matrix": m, "max_height": max_height, "max_width": max_width, "interpolation": self.interpolation}

    @classmethod
    def _expand_transform(cls, matrix, shape):
        height, width = shape
        # do not use width-1 or height-1 here, as for e.g. width=3, height=2, max_height
        # the bottom right coordinate is at (3.0, 2.0) and not (2.0, 1.0)
        rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
        dst = cv2.perspectiveTransform(np.array([rect]), matrix)[0]

        # get min x, y over transformed 4 points
        # then modify target points by subtracting these minima  => shift to (0, 0)
        dst -= dst.min(axis=0, keepdims=True)
        dst = np.around(dst, decimals=0)

        matrix_expanded = cv2.getPerspectiveTransform(rect, dst)
        max_width, max_height = dst.max(axis=0)
        return matrix_expanded, int(max_width), int(max_height)

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        pts = np.array(sorted(pts, key=lambda x: x[0]))
        left = pts[:2]  # points with smallest x coordinate - left points
        right = pts[2:]  # points with greatest x coordinate - right points

        if left[0][1] < left[1][1]:
            tl, bl = left
        else:
            bl, tl = left

        if right[0][1] < right[1][1]:
            tr, br = right
        else:
            br, tr = right

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def get_transform_init_args_names(self):
        return "scale", "keep_size", "pad_mode", "pad_val", "mask_pad_val", "fit_output", "interpolation"


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    # print(img)
    bbox = (50, 60, 50, 80)

    transform = Perspective()
    result = transform(img=img, force_apply=True, bbox=bbox)
    # print(result['img'])
    print(result["bbox"])
    show_image(result['img'])