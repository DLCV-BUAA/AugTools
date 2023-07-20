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




class LanguageModels(BaseModel):
    OPTIMIZE_ATTRIBUTES = ['external_memory', 'return_proba']

    def __init__(self, device='cpu', model_type='', temperature=1.0, top_k=100, top_p=0.01, batch_size=32,
        optimize=None, method='sentence'):
        try:
            import torch
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed torch library. Install torch by following https://pytorch.org/get-started/locally/`')
        
        super().__init__(method)
        # self.device = 'cuda' if device is None and torch.cuda.is_available() else 'cpu'
        self.device = device if device else 'cpu'
        self.model_type = model_type
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.batch_size = batch_size
        self.optimize = self.init_optimize(optimize)

    @classmethod
    def get_default_optimize_config(cls):
        return {
            'external_memory': 1024,  # GPT2 needs either zero or non-zero. XLNet needs number of extra memory tokens.
            'return_proba': False
        }

    def init_optimize(self, optimize):
        _optimize = self.get_default_optimize_config()
        if optimize is None:
            return _optimize

        for attr in self.OPTIMIZE_ATTRIBUTES:
            if attr in optimize:
                _optimize[attr] = optimize[attr]

        return _optimize

    def to(self, device):
        self.model.to(device)

    def get_device(self):
        return str(self.model.device)

    def clean(self, text):
        return text.strip()


    # for HuggingFace pipeline
    def convert_device(self, device):
        if device == 'cpu' or device is None:
            return -1
        elif device == 'cuda':
            return 0
        elif 'cuda:' in device:
            return int(device.split(':')[1])

        return -2

    @classmethod
    def control_randomness(cls, logits, seed):
        temperature = seed['temperature']
        if temperature is not None:
            return logits / temperature
        return logits

    def get_tokenizer(self):
        pass

    def get_model(self):
        pass

    def get_start_token(self):
        if self.model_type in ['bart', 'roberta']:
            return '<s>'
        if self.model_type in ['bert']:
            return '[CLS]'

    def get_separator_token(self):
        if self.model_type in ['bart', 'roberta']:
            return '</s>'
        if self.model_type in ['bert']:
            return '[SEP]'

    def get_mask_token(self):
        if self.model_type in ['bart', 'roberta', 'xlnet']:
            return '<mask>'
        if self.model_type in ['bert']:
            return '[MASK]'

    def get_pad_token(self):
        if self.model_type in ['bart', 'roberta', 'xlnet']:
            return '<pad>'
        if self.model_type in ['bert']:
            return '[PAD]'

    def get_unknown_token(self):
        if self.model_type in ['bart', 'roberta', 'xlnet']:
            return '<unk>'
        if self.model_type in ['bert']:
            return '[UNK]'

    def get_subword_prefix(self):
        if self.model_type in ['bart', 'roberta']:
            return 'Ġ'
        if self.model_type in ['xlnet']:
            return '▁'
        if self.model_type in ['bert']:
            return '##'

    def filtering(self, logits, seed):
        top_k = seed['top_k']
        top_p = seed['top_p']

        check_top_k = False
        check_top_p = False

        if top_k is not None and 0 < top_k < len(logits):
            logits, idxes = filtering.filter_top_k(logits, top_k, replace=-float('Inf'))
            check_top_k = True
        if top_p is not None and 0 < top_p < 1:
            logits, idxes = filtering.nucleus_sampling(logits, top_p)
            check_top_p = True

        # If top_p is not None, value will be sorted, so no need to select it again
        if not check_top_p:
            if check_top_k:
                logits = logits.index_select(0, idxes)
                # TODO: Externalize to util for checking
                if 'cuda' in self.device:
                    idxes = idxes.cpu()
                idxes = idxes.detach().numpy().tolist()
            else:
                idxes = np.arange(len(logits)).tolist()
        else:
            logits = logits[:len(idxes)]
            # TODO: Externalize to util for checking
            if 'cuda' in self.device:
                idxes = idxes.cpu()
            idxes = idxes.detach().numpy().tolist()

        return logits, idxes

    def pick(self, logits, idxes, target_word, n=1, include_punctuation=False):
        candidate_ids, candidate_probas = self.prob_multinomial(logits, n=n*10)
        candidate_ids = [idxes[candidate_id] for candidate_id in candidate_ids]
        results = self.get_candidates(candidate_ids, candidate_probas, target_word, n,
            include_punctuation)

        return results

    def id2token(self, _id):
        raise NotImplementedError()

    def prob_multinomial(self, logits, n):
        # Convert to probability
        probas = F.softmax(logits, dim=-1)

        # Draw candidates
        num_sample = min(n, torch.nonzero(probas, as_tuple=False).size(0))  # Number of potential candidate is small when top_k/ top_p are used.
        filtered_top_n_ids = torch.multinomial(probas, num_samples=num_sample, replacement=False).tolist()

        if self.optimize['return_proba']:
            top_n_probas = [probas[_id] for _id in filtered_top_n_ids]
            return filtered_top_n_ids, top_n_probas

        return filtered_top_n_ids, None

    def is_skip_candidate(self, candidate):
        return False

    def get_candidates(self, candidate_ids, candidate_probas, target_word=None, n=1,
        include_punctuation=False):
        # To have random behavior, NO sorting for candidate_probas.
        results = []
        if candidate_probas is None:
            candidate_probas = [0] * len(candidate_ids)

        for candidate_id, candidate_proba in zip(candidate_ids, candidate_probas):
            candidate_word = self.id2token(candidate_id)

            # unable to predict word
            if candidate_word in ['', self.get_unknown_token(), self.get_subword_prefix()] or 'unused' in candidate_word:
                continue
            # predicted same word
            if target_word is not None and candidate_word.lower() == target_word.lower():
                continue
            # stop word
            if self.is_skip_candidate(candidate_word):
                continue
            # punctuation
            if not include_punctuation and candidate_word in string.punctuation:
                continue

            results.append((candidate_word, candidate_proba))

            if len(results) >= n:
                break

        return results
    
class WordStatistics(BaseModel):
    def __init__(self, method='word'):
        super().__init__(method)


    @classmethod
    def choice(cls, x, p, size=1):
        return np.random.choice(len(x), size, p=p)
        
if __name__ == '__main__':
    model = BaseModel(method='CHAR')
    model()