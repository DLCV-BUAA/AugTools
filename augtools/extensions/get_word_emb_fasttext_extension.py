from augtools.extensions.extension import Extension
from augtools.utils.file_utils import *
from augtools.utils.text_model_utils import WordEmbeddings
from gensim.models import KeyedVectors

class GetWordFasttextModelExtension(Extension):
    def __init__(self, file_path=None, max_num_vector=None, model_name=None, top_k=100, skip_check=True, method='WORD'):
        self.method = method
        self.model_name = 'wiki-news-300d-1M' if model_name is None else model_name
        self.model = Glove(file_path, max_num_vector, self.model_name, top_k=top_k, skip_check=skip_check, method=method)
        
    def _get_rs(self, rs, **kwargs):
        rs['model'] = self.model
        return rs
 

# pre_trained_model_url = {
#     'wiki-news-300d-1M': 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
#     'wiki-news-300d-1M-subword': 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip',
#     'crawl-300d-2M': 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip',
#     'crawl-300d-2M-subword': 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip',
# }
class Fasttext(WordEmbeddings):
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
        model = KeyedVectors.load_word2vec_format(file_path, limit=max_num_vector)
        return model
    
    def _check_file_path(self, file_path=None, model_name=None, model_dim=None):
        dest_dir = file_path
        if file_path is None or not os.path.exists(file_path):
            dest_dir = DownloadUtil.get_default_dest_dir(None, 'fasttext')
            if not os.path.exists(dest_dir):
                dest_dir = DownloadUtil.download_glove(model_name)
        if model_name is None:
            model_name = 'wiki-news-300d-1M'
        file_path = os.path.join(dest_dir, '.'.join([model_name, 'vec']))
        return file_path
                
    
    
    
if __name__ == "__main__":
    
    extension = GetWordFasttextModelExtension()
    rs = None
    rs = extension(rs)
    print(rs['model'].predict('eat'))