import warnings
import random
from itertools import product

import cv2
import numpy as np

from augtools.text.transforms.word.word_transform import WordTransform
from augtools.extensions.get_word_dict_model_extension import GetWordDcitModelExtension

class WordEmbsTransform(WordTransform):
    # https://aclweb.org/anthology/D15-1306, https://arxiv.org/pdf/1804.07998.pdf, https://arxiv.org/pdf/1509.01626.pdf
    # https://arxiv.org/ftp/arxiv/papers/1812/1812.04718.pdf
    """
    Augmenter that leverage word embeddings to find top n similar word for augmentation.

    :param str model_type: Model type of word embeddings. Expected values include 'word2vec', 'glove' and 'fasttext'.
    :param str model_path: Downloaded model directory. Either model_path or model is must be provided
    :param obj model: Pre-loaded model (e.g. model class is nlpaug.model.word_embs.nmw.Word2vec(), nlpaug.model.word_embs.nmw.Glove()
        or nlpaug.model.word_embs.nmw.Fasttext())
    :param str action: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random
        position according to word embeddings calculation. If value is 'substitute', word will be replaced according
        to word embeddings calculation
    :param int top_k: Controlling lucky draw pool. Top k score token will be used for augmentation. Larger k, more
        token can be used. Default value is 100. If value is None which means using all possible tokens. This attribute will
        be ignored when using "insert" action.
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result
        from aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param bool force_reload: If True, model will be loaded every time while it takes longer time for initialization.
    :param bool skip_check: Default is False. If True, no validation for size of vocabulary embedding.
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.WordEmbsAug(model_type='word2vec', model_path='.')
    """

    def __init__(self, aug_src='glove', model_path=None, model_name=None, action='SUBSTITUTE',
        aug_min=1, aug_max=10, aug_p=0.3, top_k=100, n_gram_separator='_', model_dim=None, max_num_vector=None,
        stopwords=None, tokenizer=None, reverse_tokenizer=None, force_reload=False, stopwords_regex=None,
        skip_check=True):
        super().__init__(
            action=action, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer,
            stopwords_regex=stopwords_regex)

        self.aug_src = aug_src
        self.model_path = model_path

        self.top_k = top_k
        self.n_gram_separator = n_gram_separator

        self.model_name = model_name
        self.model_dim = model_dim
        self.skip_check = skip_check
        self.max_num_vector = max_num_vector
    
    def _append_extensions(self):
        
        return [
            GetWordDcitModelExtension(name=self.aug_src, 
                file_path=self.model_path, 
                max_num_vector=self.max_num_vector, 
                model_name=self.model_name, 
                model_dim=self.model_dim, 
                top_k=self.top_k, 
                skip_check=self.skip_check, 
                method='word'),
        ]


    def _skip_aug(self, token_idxes, tokens, model):
        results = []
        for token_idx in token_idxes:
            # Some words do not come with vector. It will be excluded in lucky draw.
            word = tokens[token_idx]

            if word in model.get_vocab():
                results.append(token_idx)

        return results

    def insert(self, data, rs=None):
        if not data or not data.strip():
            return data

        tokens = self.tokenizer(data)

        aug_idxes = self._get_random_aug_idxes(tokens, model=rs['model'])
        if not aug_idxes:
            return data
        
        aug_idxes.sort(reverse=True)

        augmented_tokens = []
        
        for aug_idx, original_token in enumerate(tokens):
            # Skip if no augment for word
            augmented_tokens.append(original_token)
            if aug_idx not in aug_idxes:
                continue

            insert_token = self.sample(rs['model'].get_vocab(), 1)[0]
            if self.n_gram_separator in insert_token:
                insert_token = insert_token.split(self.n_gram_separator)[0]

            augmented_tokens.append(insert_token)
            

        return self.reverse_tokenizer(augmented_tokens)

    def substitute(self, data, rs=None):
        if not data or not data.strip():
            return data

        tokens = self.tokenizer(data)

        aug_idxes = self._get_aug_idxes(tokens, model=rs['model'])
        if not aug_idxes:
            return data
        
        aug_idxes.sort(reverse=True)

        augmented_tokens = []
        
        for aug_idx, original_token in enumerate(tokens):
            # Skip if no augment for word
            if aug_idx not in aug_idxes:
                augmented_tokens.append(original_token)
                continue

            candidates = rs['model'].predict(original_token, n=1)
            substitute_token = self.sample(candidates, 1)[0]
            if aug_idx == 0:
                substitute_token = self._align_capitalization(original_token, substitute_token)

            augmented_tokens.append(substitute_token)
            

        return self.reverse_tokenizer(augmented_tokens)
    
    
if __name__ == '__main__':
    text = 'i eat an apple and hit someone'
    glove_transform = WordEmbsTransform(action='insert')
    tran = glove_transform(text=text,force_apply=True,n=3)
    print(text)
    print(tran['text'])  
