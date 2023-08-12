from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast, Iterable
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
        for key in rs['x']:
            if kwargs[key] is not None:
                text = kwargs[key]
                # 清洗数据    str的形式
                clean_text = self._clean_data(text)
                last_aug_text = text  
                augment_result = [last_aug_text]
                argn = kwargs.get('n', None)
                aug_num = argn if argn is not None else self.n
                action_func = self._action_select().get(self.action.lower(), None)
                max_loop = 10
                augment_result = []

                for i in range(max_loop):
                    if action_func is not None:
                        if isinstance(clean_text, list):
                            augment_result = [action_func(d, rs) for d in clean_text]
                            break
                        else:
                            augment_result += [action_func(clean_text, rs) for _ in range(aug_num)]
                            #print(augment_result)
                            augment_result = self._duplicate_augments(augment_result)
                            if len(augment_result) >= aug_num:
                                augment_result = augment_result[:aug_num]
                                break

                # TODO:去除重复增强结果   通过多个循环
                kwargs[key] = augment_result    
  
        return kwargs, rs
    
    def _duplicate_augments(self, augments):
        no_duplicate_augments = []
        for augment in augments:
            if augment not in no_duplicate_augments:
                no_duplicate_augments.append(augment)
        return no_duplicate_augments
    
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
        if isinstance(text, str):
            return text.strip()
        if isinstance(text, Iterable) :
            return [d.strip() if d else d for d in text]
        return str(text).strip()
    
    def sample(cls, x, num=None):
        if isinstance(x, list):
            return random.sample(x, num)
        elif isinstance(x, int):
            return np.random.randint(1, x-1)

    def insert(self, data, rs=None):
        raise NotImplementedError

    def substitute(self, data, rs=None):
        raise NotImplementedError

    def swap(self, data, rs=None):
        raise NotImplementedError

    def delete(self, data, rs=None):
        raise NotImplementedError

    def crop(self, data, rs=None):
        raise NotImplementedError        

    def split(self, data, rs=None):
        raise NotImplementedError
    