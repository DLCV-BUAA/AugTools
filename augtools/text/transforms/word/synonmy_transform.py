import warnings
import random
from itertools import product

import cv2
import numpy as np

from augtools.text.transforms.word.word_transform import WordTransform
from augtools.extensions.get_word_dict_model_extension import GetWordDcitModelExtension
from augtools.text.transforms.utils.part_of_speech import PartOfSpeech


class SynonmyTransform(WordTransform):
    # https://arxiv.org/pdf/1809.02079.pdf
    """
    Augmenter that leverage semantic meaning to substitute word.

    :param str lang: Language of your text. Default value is 'eng'.
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
    """

    def __init__(self, aug_min=1, aug_max=10, aug_p=0.3, lang='eng', aug_src='wordnet',
                 stopwords=None, tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, 
                ):
        super().__init__(
            action='SUBSTITUTE', aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer,stopwords_regex=stopwords_regex)

        self.aug_src = aug_src  # TODO: other source
        self.lang = lang

    def _skip_aug(self, token_idxes, tokens, model=None):
        results = []
        for token_idx in token_idxes:
            # Based on https://arxiv.org/pdf/1809.02079.pdf for Antonyms,
            # We choose only tokens which are Verbs, Adjectives, Adverbs
            if tokens[token_idx][1] not in ['VB', 'VBD', 'VBZ', 'VBG', 'VBN', 'VBP',
                'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
                continue

            # Check having antonym or not.
            # TODO: do it again in later phase. 
            if len(self.get_candidates(tokens, token_idx, model)) == 0:
                continue
            
            results.append(token_idx)

        return results

    def _get_aug_idxes(self, tokens, model=None):
        aug_cnt = self._generate_aug_cnt(len(tokens), aug_min=self.aug_min, aug_max=self.aug_max, aug_p=self.aug_p)
        word_idxes = self._pre_skip_aug(tokens, tuple_idx=0)
        word_idxes = self._skip_aug(word_idxes, tokens, model)
        if len(word_idxes) == 0:
            return None
        if len(word_idxes) < aug_cnt:
            aug_cnt = len(word_idxes)
        aug_idexes = self.sample(word_idxes, aug_cnt)
        return aug_idexes

    def get_candidates(self, tokens, token_idx, model=None):
        original_token = tokens[token_idx][0]
        word_poses = PartOfSpeech.constituent2pos(tokens[token_idx][1])
        candidates = []
        if word_poses is None or len(word_poses) == 0:
            # Use every possible words as the mapping does not defined correctly
            candidates.extend(model.predict(tokens[token_idx][0]))
        else:
            for word_pos in word_poses:
                candidates.extend(model.predict(tokens[token_idx][0], pos=word_pos))

        candidates = [c for c in candidates if c.lower() != original_token.lower()]
        return candidates

    def substitute(self, data, rs='None'):
        if not data or not data.strip():
            return data
        # 1. tokenize
        tokens = self.tokenizer(data)     
        # 2. part of speech
        pos = rs['model'].pos_tag(tokens)
        # 3. get_aug_idxes
        aug_idxes = self._get_aug_idxes(pos, rs['model'])
        
        if aug_idxes is None or len(aug_idxes) == 0:
            return data
        augmented_tokens = []

        for aug_idx, original_token in enumerate(tokens):
            # Skip if no augment for word
            if aug_idx not in aug_idxes:
                augmented_tokens.append(original_token)
                continue
            # 4. get aug idx candidate
            candidates = self.get_candidates(pos, aug_idx, rs['model'])

            if len(candidates) > 0:
                candidate = self.sample(candidates, 1)[0]
                substitute_token = candidate.replace("_", " ").replace("-", " ").lower()
                if aug_idx == 0:
                    substitute_token = self._align_capitalization(original_token, substitute_token)
                augmented_tokens.append(substitute_token)
            else:
                augmented_tokens.append(original_token)


        return self.reverse_tokenizer(augmented_tokens)

    @classmethod
    def get_model(cls, aug_src, lang):
        if aug_src == 'wordnet':
            return nmw.WordNet(lang=lang, is_synonym=True)
        elif aug_src == 'ppdb':
            return init_ppdb_model(dict_path=dict_path, force_reload=force_reload)

        raise ValueError('aug_src is not one of `wordnet` or `ppdb` while {} is passed.'.format(aug_src))

    def _append_extensions(self):
        return [
            GetWordDcitModelExtension(name=self.aug_src, lang=self.lang, is_synonym=True, method='word'),
        ]
        
        
if __name__ == '__main__':
    text = 'i eat an apple and hit someone'
    syno_transform = SynonmyTransform()
    tran = syno_transform(text=text,force_apply=True,n=3)
    print(text)
    print(tran['text'])        