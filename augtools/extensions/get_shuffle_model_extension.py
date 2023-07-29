from augtools.extensions.extension import Extension
from augtools.utils.file_utils import *
from augtools.utils.text_model_utils import BaseModel
from nltk.tokenize import sent_tokenize

class GetShuffleModelExtension(Extension):
    def __init__(self,
        mode='neighbor',
        tokenizer=None,
        method='sentence'):
        self.method = method
        self.model = Shuffle(
            mode=mode,
            tokenizer=tokenizer,
            method=method
        )
    def _get_rs(self, rs, **kwargs):
        rs['model'] = self.model
        return rs
 

class Shuffle(BaseModel):
    def __init__(self, 
        mode='neighbor',
        tokenizer=None,
        method='sentence'):
        super().__init__(method)
        self.mode = mode
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = sent_tokenize


    def tokenize(self, data):
        return self.tokenizer(data)


    def _predict_sentence(self, sentences, idx):
        last_idx = len(sentences) - 1
        direction = ''
        if self.mode == 'neighbor':
            if self.sample(2) == 0:
                direction = 'left'
            else:
                direction = 'right'
        if self.mode == 'left' or direction == 'left':
            if idx == 0:
                sentences[0], sentences[last_idx] = sentences[last_idx], sentences[0]
            else:
                sentences[idx], sentences[idx-1] = sentences[idx-1], sentences[idx]
        elif self.mode == 'right' or direction == 'right':
            if idx == last_idx:
                sentences[0], sentences[idx] = sentences[idx], sentences[0]
            else:
                sentences[idx], sentences[idx+1] = sentences[idx+1], sentences[idx]
        elif self.mode == 'random':
            idxes = self.sample(list(range(len(sentences))), num=2)
            for _id in idxes:
                if _id != idx:
                    sentences[_id], sentences[idx] = sentences[idx], sentences[_id]
                    break
        return sentences
  
    
if __name__ == "__main__":
    
    extension = GetShuffleModelExtension()
    rs = None
    rs = extension(rs)
    # print(rs['model'].predict(rs['model'].tokenize('i eat something'), 2))