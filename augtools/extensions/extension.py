from abc import ABC
from collections import defaultdict
import torch

class Extension(ABC):

    # 处理资源
    def __call__(self, rs, *args, **kwargs):
        if rs is None or len(rs) == 0:
            rs = defaultdict(dict)
        rs = self._get_rs(rs, *args, **kwargs)
        return rs

    def _get_rs(self, rs, *args, **kwargs):
        raise NotImplementedError
