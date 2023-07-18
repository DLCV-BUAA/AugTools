import warnings
import random
from itertools import product

import cv2
import numpy as np
import os

from augtools.utils.file_utils import *
from augtools.text.transforms.word.word_transform import WordTransform
from augtools.extensions.get_word_dict_model_extension import GetWordDcitModelExtension


class SpellingTransform(WordTransform):
    # https://arxiv.org/ftp/arxiv/papers/1812/1812.04718.pdf
    """
    Augmenter that leverage pre-defined spelling mistake dictionary to simulate spelling mistake.

    :param str dict_path: Path of misspelling dictionary
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.SpellingAug(dict_path='./spelling_en.txt')
    """

    def __init__(self, dict_path=None, aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, include_reverse=True, stopwords_regex=None,):
        super().__init__(
            action='SUBSTITUTE', aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, stopwords_regex=stopwords_regex)


        self.dict_path = dict_path if dict_path else os.path.join(LibraryUtil.get_res_dir(), 'text', 'word', 'spelling', 'spelling_en.txt')
        self.include_reverse = include_reverse
        self.aug_src = 'spelling'

    

    def _skip_aug(self, token_idxes, tokens, model):
        results = []
        for token_idx in token_idxes:
            # Some words do not exit. It will be excluded in lucky draw.
            token = tokens[token_idx]
            if model(token) is not None:
                results.append(token_idx)

        return results

    def substitute(self, data, rs=None):
        if not data or not data.strip():
            return data
            
        tokens = self.tokenizer(data)

        aug_idxes = self._get_aug_idxes(tokens, model=rs['model'])

        if aug_idxes is None or len(aug_idxes) == 0:
            return data

        augmented_tokens = []
        for aug_idx, original_token in enumerate(tokens):
            # Skip if no augment for word
            if aug_idx not in aug_idxes:
                augmented_tokens.append(original_token)
                continue

            candidate_words = rs['model'].predict(original_token)
            substitute_token = ''
            if candidate_words:
                substitute_token = self.sample(candidate_words, 1)[0]
            else:
                # Unexpected scenario. Adding original token
                substitute_token = original_token

            if aug_idx == 0:
                substitute_token = self._align_capitalization(original_token, substitute_token)

            augmented_tokens.append(substitute_token)
            

        return self.reverse_tokenizer(augmented_tokens)



    def _append_extensions(self):
        return [
            GetWordDcitModelExtension(name=self.aug_src, dict_path=self.dict_path, include_reverse=self.include_reverse, method='word'),
        ]

if __name__ == '__main__':
    text = 'i eat an apple and hit someone'
    syno_transform = SpellingTransform()
    tran = syno_transform(text=text,force_apply=True,n=3)
    print(text)
    print(tran['text'])   