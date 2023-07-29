import warnings
import random
from itertools import product

import cv2
import numpy as np
import os

from augtools.utils.file_utils import *
from augtools.text.transforms.word.word_transform import WordTransform
from augtools.extensions.get_word_dict_model_extension import GetWordDcitModelExtension


class SplitTransform(WordTransform):
    """
    Augmenter that apply word splitting for augmentation.

    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param int min_char: If word less than this value, do not draw word for augmentation
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter
    """

    def __init__(self, aug_min=1, aug_max=10, aug_p=0.3, stopwords=None, action='split',
                 tokenizer=None, min_char=4, reverse_tokenizer=None, stopwords_regex=None,):
        super().__init__(
            action=action, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, stopwords_regex=stopwords_regex)
        self.min_char = min_char


    def _skip_aug(self, token_idxes, tokens, model=None):
        results = []
        for token_idx in token_idxes:
            if len(tokens[token_idx]) >= self.min_char:
                results.append(token_idx)
        return results
        

    def split(self, data, rs=None):
        if not data or not data.strip():
            return data
        tokens = self._pre_process(data)
        aug_idxes = self._get_aug_idxes(tokens)
        if aug_idxes is None or len(aug_idxes) == 0:
            return data
        
        augmented_tokens = []

        for token_idx, token in enumerate(tokens):
            if token_idx not in aug_idxes:
                augmented_tokens.append(token)
                continue

            separate_pos = self.sample(len(token), 1)
            prev_token = token[:separate_pos]
            next_token = token[separate_pos:]

            augmented_tokens.append(prev_token)
            augmented_tokens.append(next_token)

        return self._post_process(augmented_tokens)



    def _pre_process(self, data=None):
        if not data or not data.strip():
                return data

        tokens = self.tokenizer(data)

        return tokens
    
    def _post_process(self, augmented_tokens):
        return self.reverse_tokenizer(augmented_tokens)




if __name__ == '__main__':
    text = 'i eat an apple and hit someone'
    random_transform = SplitTransform()
    tran = random_transform(text=text,force_apply=True,n=3)
    print(text)
    print(tran['text'])   