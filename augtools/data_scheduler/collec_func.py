import torch
import re
import collections

from augtools.core.transforms import BasicTransform
from augtools.core.compose import BaseCompose

np_str_obj_array_pattern = re.compile(r'[SaUO]')
string_classes = (str, bytes)

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


class Collection:
    def __init__(self, transforms=None):
        self.transforms = transforms
        
    def __call__(self, batch, is_x=False):
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            # print('tensor')
            if is_x:
                # print(batch, 'is_x:', is_x)
                if self.transforms is not None:
                    batch = [transform(item) for item, transform in zip(batch, self.transforms)]
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                storage = elem.storage()._new_shared(numel, device=elem.device)
                out = elem.new(storage).resize_(len(batch), *list(elem.size()))
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))
                return self.__call__([torch.as_tensor(b) for b in batch], is_x)
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int):
            # print('int', elem)
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, collections.abc.Mapping):
            try:
                return elem_type({key: self.__call__([d[key] for d in batch]) for key in elem})
            except TypeError:
                # The mapping type may not support `__init__(iterable)`.
                return {key: self.__call__([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(self.__call__(samples) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.
            # print('Sequence', transposed)
            if isinstance(elem, tuple):
                if len(transposed) == 2:
                    return [self.__call__(transposed[0], True), self.__call__(transposed[1])]
                return [self.__call__(samples) for samples in transposed]  # Backwards compatibility.
            else:
                try:
                    if len(transposed) == 2:
                        print(transposed)
                        return elem_type([self.__call__(transposed[0], True), self.__call__(transposed[1])])
                    return elem_type([self.__call__(samples) for samples in transposed])
                except TypeError:
                    # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                    return [self.__call__(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))
    
if __name__ == '__main__':
    import numpy as np
    b1 = np.array([1,2,3])
    b2 = np.array(4)

    c1 = np.array([5,6,7])
    c2 = np.array(8)

    d1 = np.array([9,10,11])
    d2 = np.array(12)

    a = [(b1, b2),(c1, c2),(d1, d2)]
    result = Collection()(a)
    print(result)

