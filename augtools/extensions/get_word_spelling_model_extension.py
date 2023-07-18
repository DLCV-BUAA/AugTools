from augtools.extensions.extension import Extension
from augtools.utils.file_utils import *
from augtools.utils.text_model_utils import BaseModel

class GetWordSpellingModelExtension(Extension):
    def __init__(self, dict_path=None, include_reverse=True, method='WORD'):
        self.method = method
        self.model = Spelling(dict_path, include_reverse, method)
    def _get_rs(self, rs, **kwargs):
        rs['model'] = self.model
        return rs
 

class Spelling(BaseModel):
    def __init__(self, dict_path=None, include_reverse=True, method='WORD'):
        super().__init__(method)
        self.include_reverse=include_reverse
        self.model = self._get_model(dict_path)

    
    def _predict_word(self, data):
        return self.model.get(data, None)
  
    def _get_model(self, dict_path=None):
        # Use default
        dict = {}
        if not dict_path:
            dict_path = os.path.join(LibraryUtil.get_res_dir(), 'text', 'word', 'spelling', 'spelling_en.txt')
        
        with open(dict_path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                tokens = line.split(' ')
                # Last token include newline separator
                tokens[-1] = tokens[-1].replace('\n', '')

                key = tokens[0]
                values = tokens[1:]

                if key not in dict:
                    dict[key] = []

                dict[key].extend(values)
                # Remove duplicate mapping
                dict[key] = list(set(dict[key]))
                # Build reverse mapping
                if self.include_reverse:
                    for value in values:
                        if value not in dict:
                            dict[value] = []
                        if key not in dict[value]:
                            dict[value].append(key)
        return dict
        
    
if __name__ == "__main__":
    
    extension = GetWordSpellingModelExtension()
    rs = None
    rs = extension(rs)
    print(rs['model']('apple'))