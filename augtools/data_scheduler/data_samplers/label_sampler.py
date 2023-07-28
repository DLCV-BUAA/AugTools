import random

from augtools.data_scheduler.data_samplers.base_sampler import BaseSampler
from augtools.data_scheduler.utils import *


class LabelSampler(BaseSampler):
    def __init__(self,
                 data_source,
                 label=None,
                 after_target_transform=False,
                 num_per_label=None,
                 num_transforms_per_instance=1,
                 ):
        """
        label : 每个符合label采样几个样例
        每个样例采样几个变换
        一共有几个label int or list 
        Args:
            data_source (_type_): _description_
            label (_type_, optional): _description_. Defaults to None.
            after_target_transform (bool, optional): _description_. Defaults to False.
        """
        self.label = label
        self.after_target_transform = after_target_transform
        self.num_per_label = num_per_label
        self.num_transforms_per_instance = num_transforms_per_instance
        super().__init__(data_source)

    def _build_index(self):
        target = getattr(self.data_source, 'targets', None)
        if target is None or self.after_target_transform:
            target = self._get_target(self.data_source)
        index = []
        if self.label is None:
            index = range(len(self.data_source))
        else:
            labels = list_obj(self.label)
            for label in labels:
                label_index = []
                for i, t in enumerate(target):
                    if t == label:
                        label_index.append(i)
                if self.num_per_label is not None:
                    min_num = min(len(label_index), self.num_per_label)
                    label_index = random.sample(label_index, min_num)
                index.extend(label_index)
        if self.num_transforms_per_instance > 0:
            index = [i for i in index for _ in range(self.num_transforms_per_instance)]
        return index

    def _get_target(self, data_source):
        samples = getattr(data_source, 'samples', None)
        if samples and not self.after_target_transform:
            targets = [s[1] for s in samples]
            return targets
        targets = []
        for sample in self.data_source:
            targets.append(sample[1])
        return targets

if __name__ == '__main__':
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(root=r'../../extensions/resource/img/foreground')
    sampler = LabelSampler(
        dataset,
        label=0,
        num_per_label=1,
        num_transforms_per_instance=2,
    )
    print(next(iter(sampler)))