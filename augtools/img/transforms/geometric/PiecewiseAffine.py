from augtools.img.transform import DualTransform
from augtools.img.transforms.utils.img_utils import *
from augtools.img.functional import ScaleFloatType
import albumentations as A
import random


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

        self.piecewise = A.PiecewiseAffine(scale, nb_rows, nb_cols, interpolation, mask_interpolation, cval,
                                           cval_mask, mode, absolute_scale, always_apply, keypoints_threshold, p)

    def __call__(self, *args, force_apply: bool = False, **kwargs):
        if (random.random() < self.p) and not self.always_apply and not force_apply:
            return kwargs
        else:
            return self.piecewise(**kwargs, force_apply=force_apply)


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    print(img.shape)
    bbox = [(170 / 500, 30 / 375, 300 / 500, 220 / 375)]
    keypoint = [(230, 80, 1, 1)]

    show_bbox_keypoint_image_float(img, bbox=bbox, keypoint=keypoint)

    transform = PiecewiseAffine()
    re = transform(image=img, force_apply=True, bboxes=bbox, keypoints=keypoint)
    show_bbox_keypoint_image_float(re['image'], bbox=re['bboxes'], keypoint=re['keypoints'])

    cc = A.PiecewiseAffine(always_apply=True)
    result = cc(image=img, bboxes=bbox, keypoints=keypoint)
    show_bbox_keypoint_image_float(result['image'], bbox=result['bboxes'], keypoint=result['keypoints'])
