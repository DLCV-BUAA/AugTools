from augtools.extensions.extension import Extension
from augtools.utils.file_utils import *
from augtools.utils.text_model_utils import BaseModel
from augtools.extensions.get_summarization_model_extension import Summarization
from augtools.extensions.get_shuffle_model_extension import Shuffle
from augtools.extensions.get_generation_model_extension import Generation
from augtools.extensions.get_lambada_model_extension import Lambada

class GetSentenceModelExtension(Extension):
    def __init__(self, name='shuffle', *arg, **kwargs):
        self.model = self.model_select(name.lower())(*arg, **kwargs)
    def _get_rs(self, rs, **kwargs):
        rs['model'] = self.model
        return rs
    
    def model_select(self, name='shuffle'):
        return {
            'shuffle': Shuffle,
            'summarization':Summarization,
            'generation': Generation,
            'lambada': Lambada,

        }.get(name, None)
    
    
if __name__ == "__main__":
    
    extension = GetSentenceModelExtension(name='shuffle')
    rs = None
    rs = extension(rs)
    print(rs['model']('apple'))