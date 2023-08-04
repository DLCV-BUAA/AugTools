import warnings
import random
from itertools import product
import math
import string

import cv2
import numpy as np

from augtools.text.transform import TextTransform
from augtools.text.transforms.utils.token import *


class WordTransform(TextTransform):
    
    def __init__(self, action, aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, swap_mode='random',
                 stopwords_regex=None, always_apply = True, p = 0.5):
        
        super().__init__(
            method='WORD', action=action, aug_min=aug_min, aug_max=aug_max, always_apply = always_apply, p = p
        )
        self.aug_p = aug_p
        self.tokenizer = tokenizer or Tokenizer.tokenizer
        self.reverse_tokenizer = reverse_tokenizer or Tokenizer.reverse_tokenizer
        self.stopwords = stopwords
        self.stopwords_regex = re.compile(stopwords_regex) if stopwords_regex else stopwords_regex

        self.swap_mode = swap_mode


    def _skip_aug(self, token_idxes, tokens, model=None):
        return token_idxes
    

    def _is_stop_words(self, token):
        return self.stopwords is not None and token in self.stopwords

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


    def _align_capitalization(self, src_token, dest_token):
        if self.get_word_case(src_token) == 'capitalize' and self.get_word_case(dest_token) == 'lower':
            return dest_token.capitalize()
        return dest_token

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
    
    def _get_aug_idxes(self, tokens, model=None):
        aug_cnt = self._generate_aug_cnt(len(tokens))
        #print(aug_cnt)
        word_idxes = self._pre_skip_aug(tokens)
        #print(word_idxes)
        word_idxes = self._skip_aug(word_idxes, tokens, model)
        #print(word_idxes)
        if len(word_idxes) == 0:
            return []
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)
        aug_idexes = self.sample(word_idxes, aug_cnt)
        #print(aug_idexes)
        return aug_idexes

    def _get_random_aug_idxes(self, tokens, model=None):
        aug_cnt = self._generate_aug_cnt(len(tokens))
        word_idxes = self._pre_skip_aug(tokens)
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)

        aug_idxes = self.sample(word_idxes, aug_cnt)

        return aug_idxes

    def _get_aug_range_idxes(self, tokens, model=None):
        aug_cnt = self._generate_aug_cnt(len(tokens))
        if aug_cnt == 0 or len(tokens) == 0:
            return []
        direction = self.sample([-1, 1], 1)[0]

        if direction > 0:
            # right
            word_idxes = [i for i, _ in enumerate(tokens[:-aug_cnt+1])]
        else:
            # left
            word_idxes = [i for i, _ in enumerate(tokens[aug_cnt-1:])]

        start_aug_idx = self.sample(word_idxes, 1)[0]
        aug_idxes = [start_aug_idx + _*direction for _ in range(aug_cnt)]

        return aug_idxes

    @classmethod
    def get_word_case(cls, word):
        if len(word) == 0:
            return 'empty'

        if len(word) == 1 and word.isupper():
            return 'capitalize'

        if word.isupper():
            return 'upper'
        elif word.islower():
            return 'lower'
        else:
            for i, c in enumerate(word):
                if i == 0:  # do not check first character
                    continue
                if c.isupper():
                    return 'mixed'

            if word[0].isupper():
                return 'capitalize'
            return 'unknown'

    def replace_stopword_by_reserved_word(self, text, stopword_reg, reserve_word):
        replaced_text = ''
        reserved_stopwords = []
    
        # pad space for easy handling
        replaced_text = ' ' + text + ' '
        for m in reversed(list(stopword_reg.finditer(replaced_text))):
            # Get position excluding prefix and suffix
            start, end, token = m.start(), m.end(), m.group()
            # replace stopword by reserve word
            replaced_text = replaced_text[:start] + reserve_word + replaced_text[end:]
            reserved_stopwords.append(token) # reversed order but it will consumed in reversed order later too
        
        # trim
        replaced_text = replaced_text[1:-1]
            
        return replaced_text, reserved_stopwords

    def replace_reserve_word_by_stopword(self, text, reserve_word_aug, original_stopwords):
        # pad space for easy handling
        replaced_text = ' ' + text + ' '
        matched = list(reserve_word_aug.finditer(replaced_text))[::-1]
        
        # TODO:?
        if len(matched) != len(original_stopwords):
            pass
        if len(matched) > len(original_stopwords):
            pass
        if len(matched) < len(original_stopwords):
            pass
        
        for m, orig_stopword in zip(matched, original_stopwords):
            # Get position excluding prefix and suffix
            start, end = m.start(), m.end()
            # replace stopword by reserve word
            replaced_text = replaced_text[:start] + orig_stopword + replaced_text[end:]
        
        # trim
        replaced_text = replaced_text[1:-1]
        
        return replaced_text

    def preprocess(self, data):
        ...

    def postprocess(self, data):
        ...
  

        
    

