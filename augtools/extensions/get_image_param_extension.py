from augtools.extensions.extension import Extension

class GetImageParamExtension(Extension):
    def _get_rs(self, rs, **kwargs):
        for key, arg in kwargs.items():
            if key in ['image', 'img', 'x']:
                rs['rows'] = arg.shape[0]
                rs['cols'] = arg.shape[1]
        return rs
    
    
if __name__ == "__main__":
    from augtools.utils.test_utils import *
    prefix = '/home/jiajunlong/Music/贾俊龙/数据增强/AugTools/augtools/img/transforms/test/'
    image = prefix + 'test.jpg'
    
    img = read_image(image)
    dicts = {'image': img}
    extension = GetImageParamExtension()
    rs = None
    rs = extension(rs, **dicts)
    print(rs['rows'])