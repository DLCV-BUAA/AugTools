from augtools.extensions.extension import Extension
from augtools.utils.file_utils import *
from augtools.utils.text_model_utils import WordStatistics

class GetWordTFIDFModelExtension(Extension):
    def __init__(self, 
        model_path=None, 
        normalize=True,
        method='WORD'
    ):
        self.model = TFIDF(
            model_path=model_path, 
            normalize=normalize,
            method=method
        )
    def _get_rs(self, rs, **kwargs):
        rs['model'] = self.model
        return rs
 

class TFIDF(WordStatistics):
    WORD_2_IDF_FILE_NAME = "tfidfaug_w2idf.txt"
    WORD_2_TFIDF_FILE_NAME = "tfidfaug_w2tfidf.txt"
    def __init__(self, 
                 model_path=None, 
                 normalize=True,
                 method='WORD'
        ):
        super().__init__(method)
        self.w2idf = {}

        self.tokens = []
        self.tfidf_scores = []
        self.w2tfidf = {}

        if model_path:
            self.read(model_path)
        self.normalize = normalize

    
    def _predict_word(self, data):
        target_idxes = self.choice(self.tokens, p=self.tfidf_scores, size=top_k)
        target_words = [self.tokens[i] for i in target_idxes]
        return target_words
    
    @classmethod
    def _normalize(cls, data):
        """
            Quoted from https://arxiv.org/pdf/1904.12848.pdf.
            // We set a high probability for replacing words with low TF-IDF scores and
            // set a low probability for replacing words with high TF-IDF scores
        """
        data = data.max() - np.copy(data)
        if data.sum() == 0:
            return [0]
        return data / data.sum()
  
    def cal_tfidf(self, word_idxes, tokens, normalize=True):
        """
            Different from traditional TF-IDF calculation, original authors treat handle single token is separately.
            Even though they are same, they will calculate TF-IDF separately. Possible reason is that they want
            to guarantee random behavior independently.
        """
        tfidf = []
        for idx in word_idxes:
            token = tokens[idx]
            tfidf.append(self.w2idf[token] / len(tokens))

        tfidf = np.array(tfidf)

        if normalize:
            return self._normalize(tfidf)

        return tfidf

    def cal_idf(self, docs_tokens):
        # Find number of documents where token t appears
        word_cnt_in_doc = {}
        for tokens in docs_tokens:
            for t in tokens:
                if t not in word_cnt_in_doc:
                    word_cnt_in_doc[t] = 0
                word_cnt_in_doc[t] += 1

        idf = {}
        for t, doc_cnt in word_cnt_in_doc.items():
            idf[t] = math.log(len(docs_tokens) / doc_cnt)

        return idf

    def train(self, data):
        self.w2idf = self.cal_idf(data)
        self.tokens = []
        self.tfidf_scores = []
        self.w2tfidf = {}

        # Build word to TF-IDF score mapping
        for tokens in data:
            for t in tokens:
                if t not in self.w2tfidf:
                    self.w2tfidf[t] = 0
                self.w2tfidf[t] += 1 / len(tokens) * self.w2idf[t]

        if self.normalize:
            tfidf_scores = list(self.w2tfidf.values())
            tfidf_scores = self._normalize(np.array(tfidf_scores))
            for i, t in enumerate(self.w2tfidf):
                self.w2tfidf[t] = tfidf_scores[i]

        self.tokens = list(self.w2tfidf.keys())
        self.tfidf_scores = list(self.w2tfidf.values())

    def save(self, model_path):
        with open(os.path.join(model_path, self.WORD_2_IDF_FILE_NAME), "w", encoding="utf-8") as f:
            for w, s in self.w2idf.items():
                f.write(str(w) + ' ' + str(s) + '\n')

        with open(os.path.join(model_path, self.WORD_2_TFIDF_FILE_NAME), "w", encoding="utf-8") as f:
            for w, s in self.w2tfidf.items():
                f.write(str(w) + ' ' + str(s) + '\n')

    def read(self, model_path):
        self.w2idf = {}
        self.w2tfidf = {}

        with open(os.path.join(model_path, self.WORD_2_IDF_FILE_NAME), 'r', encoding="utf-8") as f:
            for line in f.readlines():
                # Fix https://github.com/makcedward/nlpaug/issues/201
                try:
                    w, s = line.split(' ')
                except:
                    raise ValueError('line may include more than 1 space. Please check {}'.format(line))
                self.w2idf[w] = float(s)

        with open(os.path.join(model_path, self.WORD_2_TFIDF_FILE_NAME), 'r', encoding="utf-8") as f:
            for line in f.readlines():
                # Fix https://github.com/makcedward/nlpaug/issues/201
                try:
                    w, s = line.split(' ')
                except:
                    raise ValueError('line may include more than 1 space. Please check {}'.format(line))
                self.w2tfidf[w] = float(s)
        self.tokens = list(self.w2tfidf.keys())
        self.tfidf_scores = list(self.w2tfidf.values())

        
        
    
if __name__ == "__main__":
    
    extension = GetWordTFIDFModelExtension()
    rs = None
    rs = extension(rs)
    print(rs['model']('apple'))