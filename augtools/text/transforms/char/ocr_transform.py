import warnings
import random
from itertools import product

import cv2
import numpy as np

from augtools.text.transforms.char.char_transform import CharTransform
from augtools.extensions.get_ocr_model_extension import GetOcrModelExtension

class OcrTransform(CharTransform):
    """
    Augmenter that simulate ocr error by random values. For example, OCR may recognize I as 1 incorrectly.\
        Pre-defined OCR mapping is leveraged to replace character by possible OCR error.

    :param float aug_char_p: Percentage of character (per token) will be augmented.
    :param int aug_char_min: Minimum number of character will be augmented.
    :param int aug_char_max: Maximum number of character will be augmented. If None is passed, number of augmentation is
        calculated via aup_char_p. If calculated result from aug_char_p is smaller than aug_char_max, will use calculated result
        from aup_char_p. Otherwise, using aug_max.
    :param float aug_word_p: Percentage of word will be augmented.
    :param int aug_word_min: Minimum number of word will be augmented.
    :param int aug_word_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_word_p. If calculated result from aug_word_p is smaller than aug_word_max, will use calculated result
        from aug_word_p. Otherwise, using aug_max.
    :param int min_char: If word less than this value, do not draw word for augmentation
    :param list stopwords: List of words which will be skipped from augment operation.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param obj dict_of_path: Use pre-defined dictionary by default. Pass either file path of dict to use custom mapping. 
    :param str name: Name of this augmenter
    """

    def __init__(self, action='substitute', aug_char_min=2, aug_char_max=10, aug_char_p=0.3,
                 aug_word_p=0.3, aug_word_min=1, aug_word_max=10, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, stopwords_regex=None, min_char=1, 
                 dict_of_path=None, always_apply = True, p = 0.5):
        super().__init__(
            action=action, min_char=min_char, aug_char_min=aug_char_min, aug_char_max=aug_char_max,
            aug_char_p=aug_char_p, aug_word_min=aug_word_min, aug_word_max=aug_word_max, aug_word_p=aug_word_p,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, stopwords_regex=stopwords_regex,
            include_special_char=True, always_apply = always_apply, p = p)
        self.dict_of_path = dict_of_path
    
    def _append_extensions(self):
        return [
            GetOcrModelExtension(dict_of_path=self.dict_of_path, method='char'),
        ]



if __name__ == '__main__':
    text = 'i am 00000 years old, yean'
    ocr_transform = OcrTransform(action='delete')
    tran = ocr_transform(text=text,force_apply=True,n=3,aug_word_p=0.3)
    print(text)
    print(tran['text'])