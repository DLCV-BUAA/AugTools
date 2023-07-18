from augtools.extensions.extension import Extension
from augtools.utils.file_utils import *
from augtools.utils.text_model_utils import BaseModel

class GetOcrModelExtension(Extension):
    def __init__(self, dict_of_path=None, method='CHAR'):
        self.modmethode = method
        self.model = Ocr(dict_of_path, mode)
    def _get_rs(self, rs, **kwargs):
        rs['model'] = self.model
        return rs
 

class Ocr(BaseModel):
    def __init__(self, dict_of_path=None, method='CHAR'):
        super().__init__(method)
        self.model = self._get_model(dict_of_path)

    
    def _predict_char(self, data):
        return self.model.get(data, None)
  
    def _get_model(self, dict_of_path=None):
        # Use default
        if not dict_of_path:
            default_path = os.path.join(LibraryUtil.get_res_dir(), 'text', 'char', 'ocr', 'en.json')
            mapping = ReadUtil.read_json(default_path)
        # Use dict
        elif type(dict_of_path) is dict:
            mapping = dict_of_path
        else:
            mapping = ReadUtil.read_json(dict_of_path)
        if not mapping:
            raise ValueError('The dict_of_path does not exist. Please check "{}"'.format(dict_of_path))
        return self._generate_mapping(mapping)
    
    def _generate_mapping(self, mapping):
        result = {}

        for k in mapping:
            result[k] = mapping[k]

        # reverse mapping
        for k in mapping:
            for v in mapping[k]:
                if v not in result:
                    result[v] = []

                if k not in result[v]:
                    result[v].append(k)
        return result
    
if __name__ == "__main__":
    
    extension = GetOcrModelExtension()
    rs = None
    rs = extension(rs)
    print(rs['model']('9'))