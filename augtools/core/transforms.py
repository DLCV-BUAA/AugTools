import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
from collections import defaultdict
from warnings import warn

import cv2
import numpy as np

from augtools.utils.decorator import lists_process
# from .serialization import Serializable, get_shortest_class_fullname
# from .utils import format_args

"""
>>> image = Image()
>>> augment = NameAugmentor(p=0.5, **kargs)
>>> auged_img = augment(image)
>>> transform = Transform(
        [
            augment1,
            augment2,
        ]
    )
>>> auged = transform(image)

params: 对当前augment的设置
"""


"""
1. 准备处理数据，对数据进行变换（读图片，转换图片，处理pytorch数据，）  ===》 nparray的数据
    准备计算所需的资源
2. 定义计算方式 ===》 对图像进行计算，对标签进行计算
3. 对结果进行整理
4. 返回计算结果


思考：组合形式资源如何重复利用
"""

class BasicTransform:

    def __init__(self, always_apply: bool = False, p: float = 0.5):
        self.p = p
        self.always_apply = always_apply


    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:

        if (random.random() < self.p) or self.always_apply or force_apply:
            rs = self._prepare_rs(**kwargs)   # 准备计算所需要的资源       输出是字典形式，计算时可以通过params['name']获取
            
            kwargs, rs = self._pre_process_x(**kwargs, rs=rs) # 对数据进行处理
            kwargs, rs = self._pre_process_y(**kwargs, rs=rs) # 一般不处理？

            kwargs, rs = self._compute_x(**kwargs, rs=rs)        # 是一个确定性的操作，对输入是Tensor或者array的数据进行变换，输出是Tensor或者array的数据
            kwargs, rs = self._compute_y(**kwargs, rs=rs)

            kwargs, rs = self._post_process_x(**kwargs, rs=rs)     # 后处理，对资源进行释放，将Tensor或者array的数据按照要求进行还原
            kwargs, rs = self._post_process_y(**kwargs, rs=rs)

        return kwargs
    
    def _prepare_rs(self, **kwargs):
        """
            获取计算所需要的资源，例如图片宽高，或者背景等等
        Returns:
            资源的字典，或者资源的一个类，或者将资源作为data的一个属性，或者将资源整合到data中
        """
        exts = self._extension()
        rs = {}
        for ext in exts:
            rs = ext(rs, **kwargs)
        return rs
    
    def _pre_process_x(self, rs=None, **kwargs):
        """
            对数据进行前处理，对nlp来说，处理分词或者去除停用词等等，对图像来说，将图像转换为Tensor数据等等
        Returns:
            _type_: _description_
        """
        for key in rs['x']:
            if kwargs[key] is not None:
                func_key = 'pre_process_' + key
                target_function = self._pre_process_function_x_select.get(func_key, None)
                if target_function is not None:
                    kwargs[key], rs = target_function(kwargs[key], rs)   
        return kwargs, rs
    
    @property
    def _pre_process_function_x_select(self):
        return {
        }
        
    @property
    def _pre_process_function_y_select(self):
        return {
        }
    
    def _pre_process_y(self, rs=None, **kwargs):
        """
            对数据进行前处理，对图像来说，将图像转换为统一的格式等等
        Returns:
            _type_: _description_
        """
        for key in rs['y']:
            if kwargs[key] is not None:
                func_key = 'pre_process_' + key
                target_function = self._pre_process_function_y_select.get(func_key, None)
                if target_function is not None:
                    kwargs[key], rs = target_function(kwargs[key], rs) 
        return kwargs, rs
        
    def _compute_x(self, rs=None, **kwargs):
        """
            根据已经获取得到的资源，对数据进行处理
        Returns:
            处理后的结果
        """
        return kwargs, rs
        
    def _compute_y(self, rs=None,  **kwargs):
        """
            根据已经获取得到的资源，对数据进行处理
        Returns:
            处理后的结果
        """
        return kwargs, rs
        
    def _post_process_x(self, rs=None,  **kwargs):
        """
            对数据进行后处理，对图像来说，将图像转换为统一的格式等等
            
            
            实现：可能是rs里面存放一系列的undo操作的函数，这个函数里面只是依次执行rs的undo操作以及内存回收操作
        Returns:
            _type_: _description_
        """
        
        return kwargs, rs
    
    def _post_process_y(self, rs=None,  **kwargs):
        """
            对数据进行后处理，对图像来说，将图像转换为统一的格式等等
            实现：可能是rs里面存放一系列的undo操作的函数，这个函数里面只是依次执行rs的undo操作以及内存回收操作
        Returns:
            _type_: _description_
        """
        
        return kwargs, rs
    
    def _extension(self):
        """_summary_
            用来存放获取资源的函数项，每个函数项完成一个单独的操作，函数项可能有前后依赖关系
        Args:
            x (_type_): _description_
            rs (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # extensions = defaultdict(list)
        # extensions['x'] = [
            
        # ]
        # extensions['y'] = [
            
        # ]
        extensions = []
        return extensions



















# class DualTransform(BasicTransform):
#     """Transform for segmentation task."""

#     @property
#     def targets(self) -> Dict[str, Callable]:
#         return {
#             "image": self.apply,
#             "mask": self.apply_to_mask,
#             "masks": self.apply_to_masks,
#             "bboxes": self.apply_to_bboxes,
#             "keypoints": self.apply_to_keypoints,
#         }

#     def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
#         raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

#     def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
#         raise NotImplementedError("Method apply_to_keypoint is not implemented in class " + self.__class__.__name__)

#     def apply_to_bboxes(self, bboxes: Sequence[BoxType], **params) -> List[BoxType]:
#         return [self.apply_to_bbox(tuple(bbox[:4]), **params) + tuple(bbox[4:]) for bbox in bboxes]  # type: ignore

#     def apply_to_keypoints(self, keypoints: Sequence[KeypointType], **params) -> List[KeypointType]:
#         return [  # type: ignore
#             self.apply_to_keypoint(tuple(keypoint[:4]), **params) + tuple(keypoint[4:])  # type: ignore
#             for keypoint in keypoints
#         ]

#     def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
#         return self.apply(img, **{k: cv2.INTER_NEAREST if k == "interpolation" else v for k, v in params.items()})

#     def apply_to_masks(self, masks: Sequence[np.ndarray], **params) -> List[np.ndarray]:
#         return [self.apply_to_mask(mask, **params) for mask in masks]


# class ImageOnlyTransform(BasicTransform):
#     """Transform applied to image only."""

#     @property
#     def targets(self) -> Dict[str, Callable]:
#         return {"image": self.apply}


# class NoOp(DualTransform):
#     """Does nothing"""

#     def apply_to_keypoint(self, keypoint: KeypointInternalType, **params) -> KeypointInternalType:
#         return keypoint

#     def apply_to_bbox(self, bbox: BoxInternalType, **params) -> BoxInternalType:
#         return bbox

#     def apply(self, img: np.ndarray, **params) -> np.ndarray:
#         return img

#     def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
#         return img

#     def get_transform_init_args_names(self) -> Tuple:
#         return ()
