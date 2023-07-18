import warnings
import random
from itertools import product

import cv2
import numpy as np

from augtools.text.transforms.word.synonmy_transform import SynonmyTransform
from augtools.extensions.get_word_dict_model_extension import GetWordDcitModelExtension


class AntonymTransform(SynonmyTransform):
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
            aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords, lang=lang, aug_src=aug_src,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, stopwords_regex=stopwords_regex)



    def _append_extensions(self):
        return [
            GetWordDcitModelExtension(name=self.aug_src, lang=self.lang, is_synonym=False, method='word'),
        ]

if __name__ == '__main__':
    text = 'i eat an apple and hit someone'
    syno_transform = AntonymTransform()
    tran = syno_transform(text=text,force_apply=True,n=3)
    print(text)
    print(tran['text'])   