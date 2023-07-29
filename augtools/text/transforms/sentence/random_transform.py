"""
    Augmenter that apply operation (sentence level) to textual input based on abstractive summarization.
"""
import warnings
import random
from itertools import product

import cv2
import numpy as np
import os

from augtools.utils.file_utils import *
from augtools.text.transforms.sentence.sentence_transform import SentenceTransform
from augtools.extensions.get_sentence_model_extension import GetSentenceModelExtension


class RandomSentTransform(SentenceTransform):

    """
    Augmenter that apply randomly behavior for augmentation.

    :param str mode: Shuffle sentence to left, right, neighbor or random position. For `left`, target sentence
        will be swapped with left sentnece. For `right`, target sentence will be swapped with right sentnece.
        For `neighbor`, target sentence will be swapped with left or right sentnece radomly. For `random`, 
        target sentence will be swapped with any sentnece randomly.
    :param float aug_p: Percentage of sentence will be augmented. 
    :param int aug_min: Minimum number of sentence will be augmented.
    :param int aug_max: Maximum number of sentence will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param func tokenizer: Customize tokenization process
    :param str name: Name of this augmenter

    """

    def __init__(self, swap_mode='neighbor', action='SWAP', aug_min=1, aug_max=10, aug_p=0.3,
        tokenizer=None):
        super().__init__(
            action=action, swap_mode=swap_mode, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, tokenizer=tokenizer)
        self.aug_src = 'shuffle'

    def _append_extensions(self):
        return [
            GetSentenceModelExtension(name=self.aug_src,
                mode=self.swap_mode,
                tokenizer=self.tokenizer,
                method='sentence'
            ),
        ]

    def _pre_skip_aug(self, data):
        return list(range(len(data)))
        
    # https://arxiv.org/abs/1910.13461
    def swap(self, data, rs=None):
        if not data:
            return data

        if isinstance(data, list):
            all_data = data
        else:
            if data.strip() == '':
                return data
            all_data = [data]

        for i, d in enumerate(all_data):
            sentences = rs['model'].tokenize(d)
            aug_idxes = self._get_random_aug_idxes(sentences)
            for aug_idx in aug_idxes:
                sentences = rs['model'].predict(sentences, aug_idx)
            all_data[i] = ' '.join(sentences)

        # TODO: always return array
        if isinstance(data, list):
            return all_data
        else:
            return all_data[0]

if __name__ == '__main__':
    text = 'it is easy to say somethon but hard to do'
    random_transform = RandomSentTransform()
    tran = random_transform(text=text,force_apply=True,n=3)
    print(text)
    print(tran['text']) 

