from torch.utils.data import DataLoader

from augtools.data_scheduler.transform_generators.base_transforms import TransformGenerators
from augtools.data_scheduler.collec_func import Collection

"""
    定义数据调度器
        1. batchsize
            根据生成的样本按照batchsize进行组合
        2. label : 是否根据label进行采样
        3. transform : 每个样例进行transform的方式
        4. transform_freq : 对样本进行transform的频率,是每个样本进行一种转换，还是每个样本执行不同的转换，还是所有样本执行相同的转换
            需要一个transform生成器
            
            
        每个label一个样例?还是一个label所有样例
        
    config{
        'label': None,
        'transform': list or instance or None,
        'transform_freq': str
    }
    前提：
        1. 定义dataset
        
        ??:什么时候transform 定义collect function ? 根据collect function进行transform并且拼接数据
            根据transform generator进行generate transform
            根据generate的transform对数据进行transform
            
    实现：
        1. 定义不同的sampler选择符合config的index list
        2. 定义不同的collect function或者不同的transform generator对collect function扩展
"""


class DataScheduler:
    def __init__(self,
                 dataset,
                 batch_size=1,
                 num_workers=0,
                 pin_memory=False,
                 drop_last=False,
                 sampler=None,
                 transform=None
                 ):
        if transform is not None:
            transform = TransformGenerators(transform)
        else:
            transform = None
        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=Collection(transform),
        )

    def __getattr__(self, name):
        return self.data_loader.__getattribute__(name)

    def __iter__(self):
        return iter(self.data_loader)

    def __len__(self):
        return len(self.data_loader)

    def __next__(self):
        return next(self.data_loader)


if __name__ == '__main__':
    from augtools.data_scheduler.data_samplers.label_sampler import LabelSampler
    from torchvision.datasets import ImageFolder
    from augtools.img.transforms.blur.Brightness import Brightness
    from augtools.img.transforms.blur.Fog import FogBlur
    # from augtools.img.transforms.blur.ZoomBlur import ZoomBlur
    from augtools.img.transforms.blur.Contrast import Contrast
    import torchvision.transforms as transforms
    from augtools.utils.test_utils import show_image_by_tensor

    dataset = ImageFolder(root=r'../extensions/resource/img/foreground', transform=transforms.ToTensor())
    sampler = LabelSampler(
        dataset,
        label=[0, 1, 2],
        num_per_label=1,
        num_transforms_per_instance=3,
    )
    transform = [
        Brightness(always_apply=True),
        FogBlur(always_apply=True),
        Contrast(always_apply=True),
    ]
    data_scheduler = DataScheduler(
        dataset,
        batch_size=1,
        sampler=sampler,
        transform=transform
    )
    for i, item in enumerate(data_scheduler):
        print(i)
        show_image_by_tensor(item[0][0])
