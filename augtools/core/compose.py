from __future__ import division

import random
import typing
import warnings
from collections import defaultdict

import numpy as np
from augtools.core.transforms import BasicTransform




class BaseCompose:
    def __init__(self, transforms, p):
        if isinstance(transforms, (BaseCompose, BasicTransform)):
            warnings.warn(
                "transforms is single transform, but a sequence is expected! Transform will be wrapped into list."
            )
            transforms = [transforms]

        self.transforms = transforms
        self.p = p

    def __len__(self) -> int:
        return len(self.transforms)

    def __call__(self, *args, **data):
        raise NotImplementedError

    def __getitem__(self, item):  # type: ignore
        return self.transforms[item]



class Compose(BaseCompose):

    def __init__(
        self,
        transforms,
        bbox_params = None,
        keypoint_params = None,
        additional_targets = None,
        p = 1.0,
        is_check_shapes = True,
    ):
        super(Compose, self).__init__(transforms, p)

        # self.processors = {}
        # if bbox_params:
        #     if isinstance(bbox_params, dict):
        #         b_params = BboxParams(**bbox_params)
        #     elif isinstance(bbox_params, BboxParams):
        #         b_params = bbox_params
        #     else:
        #         raise ValueError("unknown format of bbox_params, please use `dict` or `BboxParams`")
        #     self.processors["bboxes"] = BboxProcessor(b_params, additional_targets)

        # if keypoint_params:
        #     if isinstance(keypoint_params, dict):
        #         k_params = KeypointParams(**keypoint_params)
        #     elif isinstance(keypoint_params, KeypointParams):
        #         k_params = keypoint_params
        #     else:
        #         raise ValueError("unknown format of keypoint_params, please use `dict` or `KeypointParams`")
        #     self.processors["keypoints"] = KeypointsProcessor(k_params, additional_targets)

        # if additional_targets is None:
        #     additional_targets = {}

        # self.additional_targets = additional_targets

        # for proc in self.processors.values():
        #     proc.ensure_transforms_valid(self.transforms)

        # self.add_targets(additional_targets)

        # self.is_check_args = True
        # self._disable_check_args_for_transforms(self.transforms)

        # self.is_check_shapes = is_check_shapes


    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:
        if args:
            raise KeyError("You have to pass data to augmentations as named arguments, for example: aug(image=image)")

        assert isinstance(force_apply, (bool, int)), "force_apply must have bool or int type"
        need_to_run = force_apply or random.random() < self.p
        for p in self.processors.values():
            p.ensure_data_valid(data)
        # transforms = self.transforms if need_to_run else get_always_apply(self.transforms)

        # check_each_transform = any(
        #     getattr(item.params, "check_each_transform", False) for item in self.processors.values()
        # )

        # for p in self.processors.values():
        #     p.preprocess(data)

        # for idx, t in enumerate(transforms):
        #     data = t(**data)

        #     if check_each_transform:
        #         data = self._check_data_post_transform(data)
        # data = Compose._make_targets_contiguous(data)  # ensure output targets are contiguous

        # for p in self.processors.values():
        #     p.postprocess(data)

        return data

    def _check_data_post_transform(self, data):
        rows, cols = get_shape(data["image"])

        for p in self.processors.values():
            if not getattr(p.params, "check_each_transform", False):
                continue

            for data_name in p.data_fields:
                data[data_name] = p.filter(data[data_name], rows, cols)
        return data
    

    @staticmethod
    def _make_targets_contiguous(data):
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            result[key] = value
        return result


class OneOf(BaseCompose):
    """Select one of transforms to apply. Selected transform will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

    def __init__(self, transforms, p: float = 0.5):
        super(OneOf, self).__init__(transforms, p)
        transforms_ps = [t.p for t in self.transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:

        if self.transforms_ps and (force_apply or random.random() < self.p):
            idx: int = random_utils.choice(len(self.transforms), p=self.transforms_ps)
            t = self.transforms[idx]
            data = t(force_apply=True, **data)
        return data


# class SomeOf(BaseCompose):
#     """Select N transforms to apply. Selected transforms will be called with `force_apply=True`.
#     Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.

#     Args:
#         transforms (list): list of transformations to compose.
#         n (int): number of transforms to apply.
#         replace (bool): Whether the sampled transforms are with or without replacement. Default: True.
#         p (float): probability of applying selected transform. Default: 1.
#     """

#     def __init__(self, transforms: TransformsSeqType, n: int, replace: bool = True, p: float = 1):
#         super(SomeOf, self).__init__(transforms, p)
#         self.n = n
#         self.replace = replace
#         transforms_ps = [t.p for t in self.transforms]
#         s = sum(transforms_ps)
#         self.transforms_ps = [t / s for t in transforms_ps]

#     def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:


#         if self.transforms_ps and (force_apply or random.random() < self.p):
#             idx = random_utils.choice(len(self.transforms), size=self.n, replace=self.replace, p=self.transforms_ps)
#             for i in idx:  # type: ignore
#                 t = self.transforms[i]
#                 data = t(force_apply=True, **data)
#         return data




# class OneOrOther(BaseCompose):
#     """Select one or another transform to apply. Selected transform will be called with `force_apply=True`."""

#     def __init__(
#         self,
#         first = None,
#         second = None,
#         transforms = None,
#         p: float = 0.5,
#     ):
#         if transforms is None:
#             if first is None or second is None:
#                 raise ValueError("You must set both first and second or set transforms argument.")
#             transforms = [first, second]
#         super(OneOrOther, self).__init__(transforms, p)
#         if len(self.transforms) != 2:
#             warnings.warn("Length of transforms is not equal to 2.")

#     def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:
#         if self.replay_mode:
#             for t in self.transforms:
#                 data = t(**data)
#             return data

#         if random.random() < self.p:
#             return self.transforms[0](force_apply=True, **data)

#         return self.transforms[-1](force_apply=True, **data)


# class PerChannel(BaseCompose):
#     """Apply transformations per-channel

#     Args:
#         transforms (list): list of transformations to compose.
#         channels (sequence): channels to apply the transform to. Pass None to apply to all.
#                          Default: None (apply to all)
#         p (float): probability of applying the transform. Default: 0.5.
#     """

#     def __init__(
#         self, transforms: TransformsSeqType, channels: typing.Optional[typing.Sequence[int]] = None, p: float = 0.5
#     ):
#         super(PerChannel, self).__init__(transforms, p)
#         self.channels = channels

#     def __call__(self, *args, force_apply: bool = False, **data) -> typing.Dict[str, typing.Any]:
#         if force_apply or random.random() < self.p:

#             image = data["image"]

#             # Expand mono images to have a single channel
#             if len(image.shape) == 2:
#                 image = np.expand_dims(image, -1)

#             if self.channels is None:
#                 self.channels = range(image.shape[2])

#             for c in self.channels:
#                 for t in self.transforms:
#                     image[:, :, c] = t(image=image[:, :, c])["image"]

#             data["image"] = image

#         return data



class Sequential(BaseCompose):
    """Sequentially applies all transforms to targets.

    Note:
        This transform is not intended to be a replacement for `Compose`. Instead, it should be used inside `Compose`
        the same way `OneOf` or `OneOrOther` are used. For instance, you can combine `OneOf` with `Sequential` to
        create an augmentation pipeline that contains multiple sequences of augmentations and applies one randomly
        chose sequence to input data (see the `Example` section for an example definition of such pipeline).

    Example:
        >>> transform = A.Compose([
        >>>    A.OneOf([
        >>>        A.Sequential([
        >>>            A.HorizontalFlip(p=0.5),
        >>>            A.ShiftScaleRotate(p=0.5),
        >>>        ]),
        >>>        A.Sequential([
        >>>            A.VerticalFlip(p=0.5),
        >>>            A.RandomBrightnessContrast(p=0.5),
        >>>        ]),
        >>>    ], p=1)
        >>> ])
    """

    def __init__(self, transforms, p: float = 0.5):
        super().__init__(transforms, p)

    def __call__(self, *args, **data):
        for t in self.transforms:
            data = t(**data)
        return data
