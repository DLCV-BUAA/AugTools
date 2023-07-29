import warnings
import random
from itertools import product

import cv2
import numpy as np
import os

from augtools.utils.file_utils import *
from augtools.text.transforms.word.word_transform import WordTransform
from augtools.extensions.get_word_dict_model_extension import GetWordDcitModelExtension


class RandomTransform(WordTransform):
    """
    Augmenter that apply randomly behavior for augmentation.

    :param str action: 'substitute', 'swap', 'delete' or 'crop'. If value is 'swap', adjacent words will be swapped randomly.
        If value is 'delete', word will be removed randomly. If value is 'crop', a set of contunous word will be removed randomly.
    :param float aug_p: Percentage of word will be augmented. 
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation. Not effective if action is 'crop'
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation. Not effective if action is 'crop'
    :param list target_words: List of word for replacement (used for substitute operation only). Default value is _.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter


    """

    def __init__(self, aug_min=1, aug_max=10, aug_p=0.3, stopwords=None, action='SUBSTITUTE',
                 tokenizer=None, reverse_tokenizer=None, target_words=None, stopwords_regex=None,):
        super().__init__(
            action=action, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, stopwords_regex=stopwords_regex)


        self.target_words = target_words or ['_']


    def swap(self, data, rs=None):
        tokens = self._pre_process(data)
        aug_idxes = self._get_random_aug_idxes(tokens)
        augmented_tokens = tokens
        if aug_idxes is None or len(aug_idxes) == 0 :
            return data

        for aug_idx in aug_idxes:
            swap_idx = self._get_swap_position(aug_idx, len(tokens) - 1)
            augmented_tokens = self.change_case(aug_idx, swap_idx, augmented_tokens)

        return self._post_process(augmented_tokens)

    # TODO: Tune it
    def change_case(self,original_word_idx, swap_word_idx, augmented_tokens):
        original_token = augmented_tokens[original_word_idx]
        swap_token = augmented_tokens[swap_word_idx]

        if original_word_idx != 0 and swap_word_idx != 0:
            augmented_tokens[swap_word_idx] = original_token
            augmented_tokens[original_word_idx] = swap_token
            return augmented_tokens

        original_token_case = self.get_word_case(original_token)
        swap_token_case = self.get_word_case(swap_token)

        if original_word_idx == 0:
            if original_token_case == 'capitalize':
                original_token = original_token.lower()
                
            if swap_token_case == 'lower' and original_token_case == 'capitalize':
                swap_token = swap_token.capitalize()
                

        if swap_word_idx == 0:
            if original_token_case == 'lower':
                original_token = original_token.capitalize()

            if swap_token_case == 'capitalize' and original_token_case == 'lower':
                swap_token = swap_token.lower()

        # Special case for i
        if original_token == 'i':
            original_token = 'I'
        if swap_token == 'i':
            swap_token = 'I'

        augmented_tokens[swap_word_idx] = original_token
        augmented_tokens[original_word_idx] = swap_token
        return augmented_tokens

    def _get_swap_position(self, pos, token_length):
        if pos == 0:
            # Force swap with next character if it is first character
            return pos + 1
        elif pos == token_length:
            # Force swap with previous character if it is last character
            return pos - 1
        else:
            return pos + self.sample([-1, 1], 1)[0]

    # https://arxiv.org/pdf/1703.02573.pdf, https://arxiv.org/pdf/1712.06751.pdf, https://arxiv.org/pdf/1806.09030.pdf
    # https://arxiv.org/pdf/1905.11268.pdf,
    def substitute(self, data, rs=None):
        if not data or not data.strip():
            return data
        tokens = self._pre_process(data)
        
        aug_idxes = self._get_random_aug_idxes(tokens)
        if aug_idxes is None or len(aug_idxes) == 0:
            return data
        
        augmented_tokens = tokens
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            original_token = tokens[aug_idx]
            new_token = self.sample(self.target_words, 1)[0]
            if aug_idx == 0:
                new_token = self._align_capitalization(original_token, new_token)

            augmented_tokens[aug_idx] = new_token

        return self._post_process(augmented_tokens)

    # https://arxiv.org/pdf/1905.11268.pdf, https://arxiv.org/pdf/1809.02079.pdf, https://arxiv.org/pdf/1903.09460.pdf
    def delete(self, data, rs=None):
        if not data or not data.strip():
            return data
        tokens = self._pre_process(data)
        aug_idxes = self._get_random_aug_idxes(tokens)
        if aug_idxes is None or len(aug_idxes) == 0:
            return data
        
        augmented_tokens = tokens
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            del augmented_tokens[aug_idx]

        return self._post_process(augmented_tokens)
        

    # https://github.com/makcedward/nlpaug/issues/126
    def crop(self, data, rs=None):
        if not data or not data.strip():
            return data
        tokens = self._pre_process(data)
        aug_idxes = self._get_aug_range_idxes(tokens)
        if aug_idxes is None or len(aug_idxes) == 0:
            return data
        
        augmented_tokens = tokens
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            del augmented_tokens[aug_idx]

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
    random_transform = RandomTransform()
    tran = random_transform(text=text,force_apply=True,n=3)
    print(text)
    print(tran['text'])   