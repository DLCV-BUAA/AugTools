"""
    Augmenter that apply TF-IDF based to textual input.
"""

import warnings
import random
from itertools import product
import string

import cv2
import numpy as np

from augtools.text.transforms.word.word_transform import WordTransform
from augtools.extensions.get_word_tfidf_model_extension import GetWordTFIDFModelExtension

# TODO: no test

class TfIdfTransform(WordTransform):
    # https://arxiv.org/pdf/1904.12848.pdf
    """
    Augmenter that leverage TF-IDF statistics to insert or substitute word.

    :param str model_path: Downloaded model directory. Either model_path or model is must be provided
    :param str action: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random
        position according to TF-IDF calculation. If value is 'substitute', word will be replaced according
        to TF-IDF calculation
    :param int top_k: Controlling lucky draw pool. Top k score token will be used for augmentation. Larger k, more
        token can be used. Default value is 5. If value is None which means using all possible tokens.
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result
        from aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter
    """

    def __init__(self, model_path="extensions\\resource", action='SUBSTITUTE', aug_min=1, aug_max=10, aug_p=0.3, top_k=5, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, stopwords_regex=None):
        super().__init__(
            action=action, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer,
            stopwords_regex=stopwords_regex)
        self.model_path = model_path
        self.top_k = top_k
        self.prob = random.random

        
    def _append_extensions(self):
        
        return [
            GetWordTFIDFModelExtension(model_path=self.model_path)
        ]


    def _skip_aug(self, token_idxes, tokens, model=None):
        results = []
        for token_idx in token_idxes:
            # Some word does not come with IDF. It will be excluded in lucky draw.
            word = tokens[token_idx]

            if word in model.w2idf:
                results.append(token_idx)

        return results

    def _get_aug_idxes(self, tokens, model=None):
        aug_cnt = self._generate_aug_cnt(len(tokens))
        word_idxes = self._pre_skip_aug(tokens)
        word_idxes = self._skip_aug(word_idxes, tokens, model)

        if len(word_idxes) == 0:
            return None
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)

        aug_probs = model.cal_tfidf(word_idxes, tokens)
        aug_idxes = []

        # It is possible that no token is picked. So re-try
        retry_cnt = 3
        possible_idxes = word_idxes.copy()
        for _ in range(retry_cnt):
            for i, p in zip(possible_idxes, aug_probs):
                if self.prob() < p:
                    #print(self.prob)
                    aug_idxes.append(i)
                    possible_idxes.remove(i)

                    if len(possible_idxes) == aug_cnt:
                        break

        # If still cannot pick up, random pick index regrardless probability
        if len(aug_idxes) < aug_cnt:
            aug_idxes.extend(self.sample(possible_idxes, aug_cnt-len(aug_idxes)))

        aug_idxes = self.sample(aug_idxes, aug_cnt)

        return aug_idxes

    def insert(self, data, rs=None):
        if not data or not data.strip():
            return data

        tokens = self._pre_process(data)
        aug_idxes = self._get_aug_idxes(tokens, rs['model'])
        if aug_idxes is None:
            return data

        augmented_tokens = tokens
        aug_idxes.sort(reverse=True)

        for aug_idx in aug_idxes:
            original_token = tokens[aug_idx]
            candidate_tokens = rs['model'].predict(original_token, top_k=self.top_k)
            new_token = self.sample(candidate_tokens, 1)[0]

            if aug_idx == 0:
                new_token = new_token.capitalize()

            augmented_tokens.insert(aug_idx, new_token)

        return self._post_process(augmented_tokens)

    def substitute(self, data, rs=None):
        if not data or not data.strip():
            return data
        tokens = self._pre_process(data)
        aug_idxes = self._get_aug_idxes(tokens, rs['model'])
        if aug_idxes is None:
            return data

        augmented_tokens = tokens
        for aug_idx in aug_idxes:
            original_token = tokens[aug_idx]
            candidate_tokens = rs['model'].predict(original_token, top_k=self.top_k)
            substitute_token = self.sample(candidate_tokens, 1)[0]
            if aug_idx == 0:
                substitute_token = self._align_capitalization(original_token, substitute_token)

            augmented_tokens[aug_idx] = substitute_token

        return self._post_process(augmented_tokens)
        
    def _pre_process(self, data=None):

        tokens = self.tokenizer(data)

        return tokens
    
    def _post_process(self, augmented_tokens):
        return self.reverse_tokenizer(augmented_tokens)

if __name__ == '__main__':
    text = 'i eat an apple and hit someone'
    tfidf_transform = TfIdfTransform(action='insert')
    tran = tfidf_transform(text=text,force_apply=True,n=3)
    print(text)
    print(tran['text']) 
