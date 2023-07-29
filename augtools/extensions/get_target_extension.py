from augtools.extensions.extension import Extension

class GetImageTargetExtension(Extension):
    def _get_rs(self, rs, **kwargs):
        for key, arg in kwargs.items():
            if key in ['image', 'img', 'x']:
                if rs.get('x', None) is None:
                    rs['x'] = []
                rs['x'].append(key)
            if key in ['mask', 'keypoint', 'bbox', 'masks', 'keypoints', 'bboxs',]:
                if rs.get('y', None) is None:
                    rs['y'] = []
                rs['y'].append(key)
        return rs
    
class GetTextTargetExtension(Extension):
    def _get_rs(self, rs, **kwargs):
        for key, arg in kwargs.items():
            if key in ['text', 'x']:
                if rs.get('x', None) is None:
                    rs['x'] = []
                rs['x'].append(key)
        return rs
    
    
if __name__ == "__main__":
    from augtools.utils.test_utils import *
    from collections import defaultdict
    prefix = '/home/jiajunlong/Music/贾俊龙/数据增强/AugTools/augtools/img/transforms/test/'
    image = prefix + 'test.jpg'
    
    img = read_image(image)
    dicts = {'image': img, 'mask': 5, 'keypoint': 6}
    extension = GetImageTargetExtension()
    rs = defaultdict(dict)
    rs = extension(rs, **dicts)
    print(rs)
    
   