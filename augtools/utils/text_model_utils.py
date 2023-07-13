class BaseModel:
    def __init__(self, method='CHAR'):
        self.method = method
        
    def __call__(self, data=None):
        method_func = '_predict_' + self.method.lower()
        method_func = getattr(self, method_func)
        if method_func is not None:
            return method_func(data)
        else:
            return None
        
    def _predict_char(self, data):
        # print('_predict_char')
        raise NotImplementedError  
    
    def _predict_word(self, data):
        # print('_predict_word')
        raise NotImplementedError
    
    def _predict_sentence(self, data):
        # print('_predict_sentence')
        raise NotImplementedError  
        
if __name__ == '__main__':
    model = BaseModel(method='CHAR')
    model()