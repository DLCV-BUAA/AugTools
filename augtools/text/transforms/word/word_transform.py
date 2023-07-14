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


    def _skip_aug(self, token_idxes, tokens):
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
            if _token in string.punctuation:
                continue
            if self._is_stop_words(_token):
                continue
            if self.stopwords_regex is not None and (
                    self.stopwords_regex.match(_token) or self.stopwords_regex.match(' '+_token+' ') or
                    self.stopwords_regex.match(' '+_token) or self.stopwords_regex.match(_token+' ')):
                continue
            if len(token) < self.min_char:
                continue

            results.append(token_idx)

        return results
    
    def _generate_aug_cnt(self, size, aug_min, aug_max, aug_p=None):
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

    def _get_aug_idxes(self, tokens, aug_min, aug_max, aug_p, mode):
        aug_cnt = self._generate_aug_cnt(len(tokens), aug_min, aug_max, aug_p)

        if mode == 'WORD' or mode == 'word':
            idxes = self._pre_skip_aug(tokens)
        elif mode == 'CHAR' or mode == 'char':
            idxes = [i for i, t in enumerate(tokens)]
            idxes = self._skip_aug(idxes, tokens)

        if len(idxes) == 0:
            return []
        if len(idxes) < aug_cnt:
            aug_cnt = len(idxes)
        aug_idxes = self.sample(idxes, aug_cnt)
        return aug_idxes

    def _pre_process(self, data=None):
        if not data or not data.strip():
                return data

        change_seq = 0
        data = self.tokenizer(data)

        aug_word_idxes = self._get_aug_idxes(
            data, self.aug_word_min, self.aug_word_max, self.aug_word_p, 'WORD')
        return data, aug_word_idxes
    
    def _post_process(self, result_tokens):
        return self.reverse_tokenizer(result_tokens)
    
    def _get_swap_position(self, pos, token_length, mode='adjacent'):
        if mode == 'adjacent':
            if pos == 0:
                # Force swap with next character if it is first character
                return pos + 1
            elif pos == token_length:
                # Force swap with previous character if it is last character
                return pos - 1
            else:
                return pos + self.sample([-1, 1], 1)[0]
        elif mode == 'middle':
            # Middle Random: https://arxiv.org/pdf/1711.02173.pdf
            candidates = [_ for _ in range(token_length) if _ not in [0, pos, token_length]]
            if len(candidates) == 0:
                return pos
            return self.sample(candidates, 1)[0]
        elif mode == 'random':
            # Fully Random: https://arxiv.org/pdf/1711.02173.pdf
            candidates = [_ for _ in range(token_length) if _ not in [pos]]
            if len(candidates) < 1:
                return pos
            return self.sample(candidates, 1)[0]

    def substitute(self, data, rs=None):
        tokens, aug_word_idxes = self._pre_process(data)
        if aug_word_idxes is None:
            return data

        result_tokens = []
        for token_i, token in enumerate(tokens):
            
            if token_i not in aug_word_idxes:
                result_tokens.append(token)
                continue

            chars = list(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max, self.aug_char_p,
                                                 'CHAR')
            if aug_char_idxes is None or len(aug_char_idxes) < 1:
                result_tokens.append(token)
                continue
            
            substitute_chars = []
            for char_i, char in enumerate(chars):
                if char_i not in aug_char_idxes:
                    substitute_chars.append(char)
                    continue
                candidates = rs['model'](chars[char_i])
                if candidates is not None and len(candidates) > 0:
                    substitute_chars.append(self.sample(candidates, 1)[0])
                else:
                    substitute_chars.append(char)

            # No capitalization alignment as this augmenter try to simulate random error
            new_token = ''.join(substitute_chars)
            result_tokens.append(new_token)

        return self._post_process(result_tokens)
    

    def swap(self, data, rs=None):
        tokens, aug_word_idxes = self._pre_process(data)
        if aug_word_idxes is None:
            return data

        result_tokens = []
        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                result_tokens.append(token)
                continue

            chars = list(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max, self.aug_char_p,
                                                 'CHAR')
            if aug_char_idxes is None or len(aug_char_idxes) < 1:
                result_tokens.append(token)
                continue
            
            for char_i in aug_char_idxes:
                swap_position = self._get_swap_position(char_i, len(chars)-1, mode=self.swap_mode)
                
                is_original_upper, is_swap_upper = chars[char_i].isupper(), chars[swap_position].isupper()
                original_chars = chars.copy()
                chars[char_i], chars[swap_position] = original_chars[swap_position], original_chars[char_i]

                # Swap case
                if is_original_upper:
                    chars[char_i] = chars[char_i].upper()
                else:
                    chars[char_i] = chars[char_i].lower()
                if is_swap_upper:
                    chars[swap_position] = chars[swap_position].upper()
                else:
                    chars[swap_position] = chars[swap_position].lower()

            # No capitalization alignment as this augmenter try to simulate random error

            new_token = ''.join(chars)
            result_tokens.append(new_token)

        return self._post_process(result_tokens)

    def delete(self, data, rs=None):
        tokens, aug_word_idxes = self._pre_process(data)
        if aug_word_idxes is None:
            return data

        result_tokens = []
        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                result_tokens.append(token)
                continue

            chars = list(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max, self.aug_char_p,
                                                 'CHAR')
            if aug_char_idxes is None or len(aug_char_idxes) < 1:
                result_tokens.append(token)
                continue

            aug_char_idxes.sort(reverse=True)
            for char_i in aug_char_idxes:
                del chars[char_i]

            # No capitalization alignment as this augmenter try to simulate random error

            new_token = ''.join(chars)
            result_tokens.append(new_token)

        return self._post_process(result_tokens)

    
    def insert(self, data, rs=None):
        tokens, aug_word_idxes = self._pre_process(data)
        if aug_word_idxes is None:
            return data

        result_tokens = []
        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                result_tokens.append(token)
                continue

            chars = list(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max, self.aug_char_p,
                                                 'CHAR')
            if aug_char_idxes is None or len(aug_char_idxes) < 1:
                result_tokens.append(token)
                continue

            aug_char_idxes.sort(reverse=True)
            for char_i in aug_char_idxes:
                candidates = rs['model'](chars[char_i])
                if candidates is not None and len(candidates) > 0:
                    chars.insert(char_i, self.sample(candidates, 1)[0])

            # No capitalization alignment as this augmenter try to simulate random error

            new_token = ''.join(chars)
            result_tokens.append(new_token)

        return self._post_process(result_tokens)        

        
    

