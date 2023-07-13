from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
from warnings import warn
import random
import math

import numpy as np

from augtools.core.transforms import BasicTransform  
from augtools.utils.decorator import *
from augtools.extensions.get_target_extension import *


class TextTransform(BasicTransform):
    def __init__(self, method, action, aug_min, aug_max, n=1, aug_p=0.1, always_apply = False, p = 0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.action = action
        self.method = method
        self.aug_min = aug_min
        self.aug_max = aug_max
        self.aug_p = aug_p
        self.n = n

    
    def _compute_x(self, rs=None, **kwargs):
        """
            前序操作： 清洗数据、根据初始化参数确定用来替换字符的模型
            如果data是list，增强后的data也list，每一句是只有一条增强
            如果data是str，增强后的data是self.n
            当前操作：

        """
        for key in rs['x']:
            if kwargs[key] is not None:
                text = kwargs[key]
                # 清洗数据    str的形式
                clean_text = self._clean_data(text)
                last_aug_text = text  
                augment_result = [last_aug_text]
                argn = kwargs.get('n', None)
                aug_num = argn if argn is not None else self.n
                action_func = self._action_select().get(self.action, None)
                if action_func is not None:
                    if isinstance(clean_text, list):
                        augment_result = [action_func(d, rs) for d in clean_text]
                    else:
                        augment_result = [action_func(clean_text, rs) for _ in range(aug_num)]
                    
            
                kwargs[key] = augment_result    
  
        return kwargs, rs
    
    def _extension(self):
        extensions = [
            GetTextTargetExtension(),
        ]
        append_extensions = self._append_extensions()
        if len(append_extensions) > 0:
            extensions.extend(append_extensions)
        return extensions
    
    def _append_extensions(self):
        return []
    
    def _action_select(self):
        return {
            'insert': self.insert,
            'substitute': self.substitute,
            'swap': self.swap,
            'delete': self.delete,
            'crop': self.crop,
            'split': self.split,
        }
    
    def _clean_data(cls, text):
        if isinstance(text, list) :
            return [d.strip() for d in text]
        return text.strip()
    
    def sample(cls, x, num=None):
        if isinstance(x, list):
            return random.sample(x, num)
        elif isinstance(x, int):
            return np.random.randint(1, x-1)

    def insert(self, data):
        raise NotImplementedError

    def substitute(self, data):
        raise NotImplementedError

    def swap(self, data):
        raise NotImplementedError

    def delete(self, data):
        raise NotImplementedError

    def crop(self, data):
        raise NotImplementedError        

    def split(self, data):
        raise NotImplementedError
    
    

