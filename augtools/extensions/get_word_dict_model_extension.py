from augtools.extensions.extension import Extension
from augtools.utils.file_utils import *
from augtools.utils.text_model_utils import BaseModel
import nltk
from nltk.corpus import wordnet
from augtools.extensions.get_wordnet_model_extension import WordNet
from augtools.extensions.get_word_spelling_model_extension import Spelling
from augtools.extensions.get_word_emb_glove_extension import Glove
from augtools.extensions.get_machine_translation_model_extension import MtModels
from augtools.extensions.get_word_emb_word2vec_extension import Word2vec
from augtools.extensions.get_word_emb_fasttext_extension import Fasttext

class GetWordDcitModelExtension(Extension):
    def __init__(self, name='wordnet', *arg, **kwargs):
        self.model = self.model_select(name.lower())(*arg, **kwargs)
    def _get_rs(self, rs, **kwargs):
        rs['model'] = self.model
        return rs
    
    def model_select(self, name='wordnet'):
        return {
            'wordnet': WordNet,
            'spelling': Spelling,
            'glove': Glove,
            'machine_translation': MtModels,
            'word2vec': Word2vec,
            'fasttext': Fasttext,
            
        }.get(name, None)
    
    
if __name__ == "__main__":
    
    extension = GetWordDcitModelExtension(name='wordnet', lang='eng')
    rs = None
    rs = extension(rs)
    print(rs['model']('apple'))