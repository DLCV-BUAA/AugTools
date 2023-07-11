from augtools.utils.test_utils import *
from functools import wraps
import numpy as np

def lists_process(func):
    def wrapped_function(self, data, *args, **kwargs):
        result = None
        # print(data)
        if isinstance(data, list):
            return [func(self, d, *args, **kwargs) for d in data]
        else:
            return func(data, *args, **kwargs) 
    return wrapped_function



def preserve_shape(func):
    """Preserve shape of the image"""

    @wraps(func)
    def wrapped_function(img: np.ndarray, *args, **kwargs) -> np.ndarray:
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function


def process_in_chunks(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        num_channels = img.shape[2] if len(img.shape) == 3 else 1
        if num_channels > 4:
            chunks = []
            for index in range(0, num_channels, 4):
                if num_channels - index == 2:
                    for i in range(2):
                        chunk = img[:, :, index + i : index + i + 1]
                        chunk = func(chunk, **kwargs)
                        chunk = np.expand_dims(chunk, -1)
                        chunks.append(chunk)
                else:
                    chunk = img[:, :, index : index + 4]
                    chunk = func(chunk, *args, **kwargs)
                    chunks.append(chunk)
            img = np.dstack(chunks)
        else:
            img = func(img, *args, **kwargs)
        return img

    return wrapped_function