from torch.utils.data import DataLoader, Dataset, SequentialSampler

class BaseSampler(SequentialSampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self._index = self._build_index()

    def __iter__(self):
        return iter(self._index)

    def __len__(self):
        return len(self._index)
    
        
    def _build_index(self):
        return range(len(self.data_source))
