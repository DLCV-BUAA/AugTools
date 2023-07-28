from itertools import cycle

class TransformGenerators:
    def __init__(self, transforms):
        if transforms is None:
            self.transforms = None
        if isinstance(transforms, list):
            self.transforms = transforms
        else:
            self.transforms = [transforms]
        if transforms is not None:
            self.transforms = self._build_iter(self.transforms)
            self.transforms = cycle(self.transforms)
    
    # def __next__(self):
    #     next(self.transforms)
    
    def __iter__(self):
        return self.transforms
    
    def _build_iter(self, transforms):
        return transforms

if __name__ == '__main__':
    from augtools.img.transforms.blur.Fog import FogBlur
    from augtools.img.transforms.blur.Brightness import Brightness
    transformers = TransformGenerators(transforms=[FogBlur(), Brightness()])
    for i in range(10):
        print(next(iter(transformers)))

            
