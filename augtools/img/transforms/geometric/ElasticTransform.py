from augtools.img.transform import DualTransform
from augtools.img.transforms.utils.img_utils import *
import random
from augtools.img import functional as F
import albumentations as A


class ElasticTransform(DualTransform):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

    Args:
        alpha (float):
        sigma (float): Gaussian filter parameter.
        alpha_affine (float): The range will be (-alpha_affine, alpha_affine)
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        border_mode (OpenCV flag): flag that is used to specify the pixel extrapolation method. Should be one of:
            cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101.
            Default: cv2.BORDER_REFLECT_101
        value (int, float, list of ints, list of float): padding value if border_mode is cv2.BORDER_CONSTANT.
        mask_value (int, float,
                    list of ints,
                    list of float): padding value if border_mode is cv2.BORDER_CONSTANT applied for masks.
        approximate (boolean): Whether to smooth displacement map with fixed kernel size.
                               Enabling this option gives ~2X speedup on large images.
        same_dxdy (boolean): Whether to use same random generated shift for x and y.
                             Enabling this option gives ~2X speedup.

    Targets:
        image, mask, bbox

    Image types:
        uint8, float32
    """

    def __init__(
        self,
        alpha=1,
        sigma=50,
        alpha_affine=50,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        always_apply=False,
        approximate=False,
        same_dxdy=False,
        p=0.5,
    ):
        super(ElasticTransform, self).__init__(always_apply, p)
        self.alpha = alpha
        self.alpha_affine = alpha_affine
        self.sigma = sigma
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.approximate = approximate
        self.same_dxdy = same_dxdy

    def __call__(self, *args, force_apply: bool = False, **kwargs):
        if (random.random() < self.p) and not self.always_apply and not force_apply:
            return kwargs

        res = {}
        random_state = random.randint(0, 10000)
        h, w, _ = kwargs["img"].shape

        if "img" in kwargs:
            res["img"] = F.elastic_transform(
                            img,
                            self.alpha,
                            self.sigma,
                            self.alpha_affine,
                            self.interpolation,
                            self.border_mode,
                            self.value,
                            np.random.RandomState(random_state),
                            self.approximate,
                            self.same_dxdy,
                            )

        if "bbox" in kwargs:
            bboxes = []
            for item in kwargs["bbox"]:
                bboxes.append(self.apply_to_bbox(item, random_state=random_state, h=h, w=w))
            res["bbox"] = bboxes

        return res

    def apply_to_bbox(self, bbox, random_state=None, h=None, w=None):
        rows, cols = h, w
        mask = np.zeros((rows, cols), dtype=np.uint8)
        bbox_denorm = F.denormalize_bbox(bbox, rows, cols)
        x_min, y_min, x_max, y_max = bbox_denorm[:4]
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        mask[y_min:y_max, x_min:x_max] = 1
        mask = F.elastic_transform(
            mask,
            self.alpha,
            self.sigma,
            self.alpha_affine,
            cv2.INTER_NEAREST,
            self.border_mode,
            self.mask_value,
            np.random.RandomState(random_state),
            self.approximate,
        )
        bbox_returned = F.bbox_from_mask(mask)
        bbox_returned = F.normalize_bbox(bbox_returned, rows, cols)
        return bbox_returned


if __name__ == '__main__':
    from augtools.utils.test_utils import *

    prefix = f'../test/'
    image = prefix + 'test.jpg'

    img = read_image(image)
    print(img.shape)
    bbox = [(170 / 500, 30 / 375, 300 / 500, 220 / 375)]
    keypoint = [(230, 80, 1, 1)]

    show_bbox_keypoint_image_float(img, bbox=bbox, keypoint=keypoint)

    transform = ElasticTransform()
    re = transform(img=img, force_apply=True, bbox=bbox, keypoint=keypoint)
    show_bbox_keypoint_image_float(re['img'], bbox=re['bbox'], keypoint=None)

    cc = A.ElasticTransform(always_apply=True)
    result = cc(image=img, bboxes=bbox)
    show_bbox_keypoint_image_float(result['image'], bbox=result['bboxes'], keypoint=None)
