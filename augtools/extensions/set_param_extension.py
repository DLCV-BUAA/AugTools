from augtools.extensions.extension import Extension

class SetParamExtension(Extension):
    def __init__(self, param):
        self.param = param
    def _get_rs(self, rs, **kwargs):
        rs.update(self.param)
        return rs
    
    
if __name__ == "__main__":
    # from augtools.utils.test_utils import *
    # prefix = '/home/jiajunlong/Music/贾俊龙/数据增强/AugTools/augtools/img/transforms/test/'
    # image = prefix + 'test.jpg'
    
    # img = read_image(image)
    dicts = {'image': 5, 'b':6}
    extension = SetParamExtension(dicts)
    rs = None
    rs = extension(rs)
    print(rs)