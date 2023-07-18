from augtools.extensions.extension import Extension
from augtools.utils.file_utils import *
from augtools.utils.text_model_utils import WordEmbeddings
from gensim.models import KeyedVectors

class GetWordGloveModelExtension(Extension):
    def __init__(self, file_path=None, max_num_vector=None, model_name=None, top_k=100, skip_check=True, method='WORD'):
        self.method = method
        self.model_name = 'glove.6B' if model_name is None else model_name
        self.model = Glove(file_path, max_num_vector, self.model_name, top_k=top_k, skip_check=skip_check, method=method)
        
    def _get_rs(self, rs, **kwargs):
        rs['model'] = self.model
        return rs
 

# pre_trained_model_url = {
#     'glove.6b': 'http://nlp.stanford.edu/data/glove.6B.zip',
#     'glove.42b.300d': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
#     'glove.840b.300d': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
#     'glove.twitter.27b': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
# }
class Glove(WordEmbeddings):
    def __init__(self, file_path=None, max_num_vector=None, model_name=None, model_dim=None, top_k=100, skip_check=True , method='WORD'):
        super().__init__(top_k=100, skip_check=True , method='WORD')
        file_path = self._check_file_path(file_path, model_name, model_dim)
        self.model = self._get_model(file_path, max_num_vector)
        super()._init()

    
    def _predict_word(self, data, n=1):
        result = self.model.most_similar(data, topn=self.top_k+1)
        result = [w for w, s in result if w.lower() != data.lower()]
        return result[:self.top_k]
  
    def _get_model(self, file_path=None, max_num_vector=None):
        model = KeyedVectors.load_word2vec_format(file_path, binary=False, no_header=True, limit=max_num_vector)
        return model
    
    def _check_file_path(self, file_path=None, model_name=None, model_dim=None):
        dest_dir = file_path
        if file_path is None or not os.path.exists(file_path):
            dest_dir = DownloadUtil.get_default_dest_dir(None, 'glove')
            if not os.path.exists(dest_dir):
                dest_dir = DownloadUtil.download_glove(model_name)
        if model_name is None:
            model_name = 'glove.6B'
        if model_dim is None:
            model_dim = 50
        file_path = os.path.join(dest_dir, '.'.join([model_name, str(model_dim)+'d', 'txt']))
        return file_path
                
    
    
    
if __name__ == "__main__":
    
    extension = GetWordGloveModelExtension()
    rs = None
    rs = extension(rs)
    print(rs['model'].predict('eat'))