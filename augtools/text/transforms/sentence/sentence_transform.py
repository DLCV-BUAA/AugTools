import warnings
import random
from itertools import product
import math
import string

import cv2
import numpy as np

from augtools.text.transform import TextTransform
from augtools.text.transforms.utils.token import *


class SentenceTransform(TextTransform):
    
    def __init__(self, action, aug_min=1, aug_max=10, aug_p=0.3, stopwords=None, swap_mode='random',
                 tokenizer=None, reverse_tokenizer=None, device='cuda', always_apply = True, p = 0.5):
        
        super().__init__(
            method='sentence', action=action, aug_min=aug_min, aug_max=aug_max, always_apply = always_apply, p = p
        )
        self.aug_p = aug_p
        self.tokenizer = tokenizer or Tokenizer.tokenizer
        self.stopwords = stopwords
        self.device = device
        self.swap_mode = swap_mode
        
    def _is_duplicate(cls, dataset, data):
        for d in dataset:
            if d == data:
                return True
        return False  
    
    def _generate_aug_cnt(self, size, aug_min=1, aug_max=10, aug_p=None):
        if size == 0:
            return 0
        if aug_p is not None:
            percent = aug_p
        elif self.aug_p:
            percent = self.aug_p
        else:
            percent = 0.3
        cnt = int(math.ceil(percent * size))

        if aug_min and cnt < aug_min:
            return aug_min
        if aug_max and cnt > aug_max:
            return aug_max
        return cnt 
    
    def _get_random_aug_idxes(self, tokens, model=None):
        aug_cnt = self._generate_aug_cnt(len(tokens))
        word_idxes = self._pre_skip_aug(tokens)
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)

        aug_idxes = self.sample(word_idxes, aug_cnt)

        return aug_idxes 
    
    def _pre_skip_aug(self, tokens, tuple_idx=None):
        results = []
        for token_idx, token in enumerate(tokens):
            if tuple_idx is not None:
                _token = token[tuple_idx]
            else:
                _token = token
            # skip punctuation
            if _token in string.punctuation:
                continue
            # skip stopwords by list
            if self._is_stop_words(_token):
                continue
            # skip stopwords by regex
            # https://github.com/makcedward/nlpaug/issues/81
            if self.stopwords_regex is not None and (
                    self.stopwords_regex.match(_token) or self.stopwords_regex.match(' '+_token+' ') or
                    self.stopwords_regex.match(' '+_token) or self.stopwords_regex.match(_token+' ')):
                continue

            results.append(token_idx)

        return results


    
