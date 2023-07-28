from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
from warnings import warn

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from augtools.core.transforms import BasicTransform
from augtools.utils.decorator import *
from augtools.extensions.get_image_param_extension import GetImageParamExtension
from augtools.extensions.get_target_extension import GetImageTargetExtension

NumType = Union[int, float, np.ndarray]
BoxInternalType = Tuple[float, float, float, float]
BoxType = Union[BoxInternalType, Tuple[float, float, float, float, Any]]
KeypointInternalType = Tuple[float, float, float, float]
KeypointType = Union[KeypointInternalType, Tuple[float, float, float, float, Any]]
ImageColorType = Union[float, Sequence[float]]

ScaleFloatType = Union[float, Tuple[float, float]]
ScaleIntType = Union[int, Tuple[int, int]]

FillValueType = Optional[Union[int, float, Sequence[int], Sequence[float]]]


class ImageTransform(BasicTransform):
    def _compute_x(self, rs=None, **kwargs):
        for key in rs['x']:
            if kwargs[key] is not None:
                img = kwargs[key]
                img = self._compute_x_function(img, rs=rs)
                kwargs[key] = img
        return kwargs, rs

    def _extension(self):
        extensions = [
            GetImageTargetExtension()
        ]
        append_extensions = self._append_extensions()
        if len(append_extensions) > 0:
            extensions.extend(append_extensions)
        return extensions

    def _append_extensions(self):
        return []

    def _compute_x_function(self, img, rs=None):
        raise NotImplementedError

    def _pre_process_x(self, rs=None, **kwargs):
        """
            如果数据是Tensor类型，转换为ndarray，或者Image.Image类型
        Returns:
            _type_: _description_
        """
        for key in rs['x']:
            if kwargs[key] is not None:
                if torch.is_tensor(kwargs[key]):
                    kwargs[key] = transforms.ToPILImage()(kwargs[key])
                    kwargs[key] = np.array(kwargs[key])
                elif isinstance(kwargs[key], Image.Image):
                    kwargs[key] = np.array(kwargs[key])
                func_key = 'pre_process_' + key
                target_function = self._pre_process_function_x_select.get(func_key, None)
                if target_function is not None:
                    kwargs[key], rs = target_function(kwargs[key], rs)
        return kwargs, rs


class DualTransform(ImageTransform):

    def _compute_y(self, rs=None, **kwargs):
        for key in rs['y']:
            if kwargs[key] is not None:
                y_rs = kwargs[key]
                y_rs = self._compute_function_y_select.get(key)(kwargs[key], rs)
                kwargs[key] = y_rs
        return kwargs, rs

    @property
    def _compute_function_y_select(self):
        return {
            'bbox': self._compute_bbox_function,
            'bboxs': self._compute_bbox_function,

            'keypoint': self._compute_keypoint_function,
            'keypoints': self._compute_keypoint_function,

            'mask': self._compute_mask_function,
            'masks': self._compute_mask_function,
        }

    @lists_process
    def _compute_bbox_function(self, y, rs=None):
        raise NotImplementedError

    @lists_process
    def _compute_keypoint_function(self, y, rs=None):
        raise NotImplementedError

    @lists_process
    def _compute_mask_function(self, y, rs=None):
        return self._compute_x_function(y, rs)
