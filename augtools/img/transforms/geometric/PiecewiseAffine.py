from augtools.img.transform import DualTransform
from augtools.img.transforms.utils.img_utils import *
import skimage as sk
import random
from typing_extensions import Concatenate, ParamSpec
import math

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



class PiecewiseAffine(DualTransform):
    """Apply affine transformations that differ between local neighbourhoods.
    This augmentation places a regular grid of points on an image and randomly moves the neighbourhood of these point
    around via affine transformations. This leads to local distortions.

    This is mostly a wrapper around scikit-image's ``PiecewiseAffine``.
    See also ``Affine`` for a similar technique.

    Note:
        This augmenter is very slow. Try to use ``ElasticTransformation`` instead, which is at least 10x faster.

    Note:
        For coordinate-based inputs (keypoints, bounding boxes, polygons, ...),
        this augmenter still has to perform an image-based augmentation,
        which will make it significantly slower and not fully correct for such inputs than other transforms.

    Args:
        scale (float, tuple of float): Each point on the regular grid is moved around via a normal distribution.
            This scale factor is equivalent to the normal distribution's sigma.
            Note that the jitter (how far each point is moved in which direction) is multiplied by the height/width of
            the image if ``absolute_scale=False`` (default), so this scale can be the same for different sized images.
            Recommended values are in the range ``0.01`` to ``0.05`` (weak to strong augmentations).
                * If a single ``float``, then that value will always be used as the scale.
                * If a tuple ``(a, b)`` of ``float`` s, then a random value will
                  be uniformly sampled per image from the interval ``[a, b]``.
        nb_rows (int, tuple of int): Number of rows of points that the regular grid should have.
            Must be at least ``2``. For large images, you might want to pick a higher value than ``4``.
            You might have to then adjust scale to lower values.
                * If a single ``int``, then that value will always be used as the number of rows.
                * If a tuple ``(a, b)``, then a value from the discrete interval
                  ``[a..b]`` will be uniformly sampled per image.
        nb_cols (int, tuple of int): Number of columns. Analogous to `nb_rows`.
        interpolation (int): The order of interpolation. The order has to be in the range 0-5:
             - 0: Nearest-neighbor
             - 1: Bi-linear (default)
             - 2: Bi-quadratic
             - 3: Bi-cubic
             - 4: Bi-quartic
             - 5: Bi-quintic
        mask_interpolation (int): same as interpolation but for mask.
        cval (number): The constant value to use when filling in newly created pixels.
        cval_mask (number): Same as cval but only for masks.
        mode (str): {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
            Points outside the boundaries of the input are filled according
            to the given mode.  Modes match the behaviour of `numpy.pad`.
        absolute_scale (bool): Take `scale` as an absolute value rather than a relative value.
        keypoints_threshold (float): Used as threshold in conversion from distance maps to keypoints.
            The search for keypoints works by searching for the
            argmin (non-inverted) or argmax (inverted) in each channel. This
            parameters contains the maximum (non-inverted) or minimum (inverted) value to accept in order to view a hit
            as a keypoint. Use ``None`` to use no min/max. Default: 0.01

    Targets:
        image, mask, keypoints, bboxes

    Image types:
        uint8, float32

    """

    def __init__(
        self,
        scale: ScaleFloatType = (0.03, 0.05),
        nb_rows: Union[int, Sequence[int]] = 4,
        nb_cols: Union[int, Sequence[int]] = 4,
        interpolation: int = 1,
        mask_interpolation: int = 0,
        cval: int = 0,
        cval_mask: int = 0,
        mode: str = "constant",
        absolute_scale: bool = False,
        always_apply: bool = False,
        keypoints_threshold: float = 0.01,
        p: float = 0.5,
    ):
        super(PiecewiseAffine, self).__init__(always_apply, p)

        self.scale = to_tuple(scale, scale)
        self.nb_rows = to_tuple(nb_rows, nb_rows)
        self.nb_cols = to_tuple(nb_cols, nb_cols)
        self.interpolation = interpolation
        self.mask_interpolation = mask_interpolation
        self.cval = cval
        self.cval_mask = cval_mask
        self.mode = mode
        self.absolute_scale = absolute_scale
        self.keypoints_threshold = keypoints_threshold

    def get_transform_init_args_names(self):
        return (
            "scale",
            "nb_rows",
            "nb_cols",
            "interpolation",
            "mask_interpolation",
            "cval",
            "cval_mask",
            "mode",
            "absolute_scale",
            "keypoints_threshold",
        )

    @property
    def targets_as_params(self):
        return ["image"]

    def get_params_dependent_on_targets(self, params) -> dict:
        h, w = params["image"].shape[:2]

        nb_rows = np.clip(random.randint(*self.nb_rows), 2, None)
        nb_cols = np.clip(random.randint(*self.nb_cols), 2, None)
        nb_cells = nb_cols * nb_rows
        scale = random.uniform(*self.scale)

        jitter: np.ndarray = random_utils.normal(0, scale, (nb_cells, 2))
        if not np.any(jitter > 0):
            return {"matrix": None}

        y = np.linspace(0, h, nb_rows)
        x = np.linspace(0, w, nb_cols)

        # (H, W) and (H, W) for H=rows, W=cols
        xx_src, yy_src = np.meshgrid(x, y)

        # (1, HW, 2) => (HW, 2) for H=rows, W=cols
        points_src = np.dstack([yy_src.flat, xx_src.flat])[0]

        if self.absolute_scale:
            jitter[:, 0] = jitter[:, 0] / h if h > 0 else 0.0
            jitter[:, 1] = jitter[:, 1] / w if w > 0 else 0.0

        jitter[:, 0] = jitter[:, 0] * h
        jitter[:, 1] = jitter[:, 1] * w

        points_dest = np.copy(points_src)
        points_dest[:, 0] = points_dest[:, 0] + jitter[:, 0]
        points_dest[:, 1] = points_dest[:, 1] + jitter[:, 1]

        # Restrict all destination points to be inside the image plane.
        # This is necessary, as otherwise keypoints could be augmented
        # outside of the image plane and these would be replaced by
        # (-1, -1), which would not conform with the behaviour of the other augmenters.
        points_dest[:, 0] = np.clip(points_dest[:, 0], 0, h - 1)
        points_dest[:, 1] = np.clip(points_dest[:, 1], 0, w - 1)

        matrix = skimage.transform.PiecewiseAffineTransform()
        matrix.estimate(points_src[:, ::-1], points_dest[:, ::-1])

        return {
            "matrix": matrix,
        }

    def apply(self, img: np.ndarray, matrix: skimage.transform.PiecewiseAffineTransform = None, **params) -> np.ndarray:
        return F.piecewise_affine(img, matrix, self.interpolation, self.mode, self.cval)

    def apply_to_mask(
        self, img: np.ndarray, matrix: skimage.transform.PiecewiseAffineTransform = None, **params
    ) -> np.ndarray:
        return F.piecewise_affine(img, matrix, self.mask_interpolation, self.mode, self.cval_mask)

    def apply_to_bbox(
        self,
        bbox: BoxInternalType,
        rows: int = 0,
        cols: int = 0,
        matrix: skimage.transform.PiecewiseAffineTransform = None,
        **params
    ) -> BoxInternalType:
        return F.bbox_piecewise_affine(bbox, matrix, rows, cols, self.keypoints_threshold)

    def apply_to_keypoint(
        self,
        keypoint: KeypointInternalType,
        rows: int = 0,
        cols: int = 0,
        matrix: skimage.transform.PiecewiseAffineTransform = None,
        **params
    ):
        return F.keypoint_piecewise_affine(keypoint, matrix, rows, cols, self.keypoints_threshold)

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