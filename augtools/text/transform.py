from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
from warnings import warn

import numpy as np

from augtools.core.transforms import BasicTransform  
from augtools.utils.decorator import *


class TextTransform(BasicTransform):
    @property
    def _pre_process_function_x_select(self):
        prefix = 'pre_process_'
        return {
            prefix + 'text': self._pre_process_text,
            prefix + 'x' : self._pre_process_text,
        }
    
    def _pre_process_text(self, text, rs):
        
        return text, rs