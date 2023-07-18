import augtools.utils.text_model_utils as normalization
class BaseModel:
    def __init__(self, method='CHAR'):
        self.method = method
        
    def __call__(self, data=None, *args, **kwargs):
        method_func = '_predict_' + self.method.lower()
        method_func = getattr(self, method_func)
        if method_func is not None:
            return method_func(data, *args, **kwargs)
        else:
            return None
        
    def predict(self,*args, **kwargs):
        return self.__call__(*args, **kwargs)
        
    def _predict_char(self, data, *args, **kwargs):
        # print('_predict_char')
        raise NotImplementedError  
    
    def _predict_word(self, data, *args, **kwargs):
        # print('_predict_word')
        raise NotImplementedError
    
    def _predict_sentence(self, data, *args, **kwargs):
        # print('_predict_sentence')
        raise NotImplementedError  
    
    
class WordEmbeddings(BaseModel):
    def __init__(self, top_k=100, skip_check=True, method='WORD'):
        super().__init__(method)
        self.top_k = top_k
        self.skip_check = skip_check
        self.emb_size = 0
        self.vocab_size = 0
        self.words = []


    def _init(self):
        self.words = [self.model.index_to_key[i] for i in range(len(self.model.index_to_key))]
        self.emb_size = self.model[self.model.key_to_index[self.model.index_to_key[0]]]
        self.vocab_size = len(self.words)

    def download(self, model_path):
        raise NotImplementedError

    def get_vocab(self):
        return self.words

    @classmethod
    def _normalize(cls, vectors, norm='l2'):
        if norm == 'l2':
            return normalization.l2_norm(vectors)
        elif norm == 'l1':
            return normalization.l1_norm(vectors)
        elif norm == 'standard':
            return normalization.standard_norm(vectors)


        
if __name__ == '__main__':
    model = BaseModel(method='CHAR')
    model()