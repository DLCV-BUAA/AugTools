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


class AbstSummSentTransform(SentenceTransform):

    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_path: Model name or model path. It used transformers to load the model. Tested 'facebook/bart-large-cnn',
        t5-small', 't5-base' and 't5-large'. For models, you can visit https://huggingface.co/models?filter=summarization
    :param int batch_size: Batch size.
    :param int min_length: The min length of output text.
    :param int max_length: The max length of output text. 
    :param float temperature: The value used to module the next token probabilities.
    :param int top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
    :param float top_p: If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or
        higher are kept for generation.
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param str name: Name of this augmenter

    """

    def __init__(self, model_path='t5-base', tokenizer_path='t5-base', device='cuda', action='substitute',
        min_length=20, max_length=50, batch_size=32, temperature=1.0, top_k=50, top_p=0.9,):
        super().__init__(
            action=action, device=device)
        self.aug_src = 'summarization'
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.min_length = min_length
        self.max_length = max_length
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

    def _append_extensions(self):
        return [
            GetSentenceModelExtension(name=self.aug_src,
                model_name=self.model_path, 
                tokenizer_name=self.tokenizer_path, 
                min_length=self.min_length, 
                device=self.device, 
                max_length=self.max_length,
                batch_size=self.batch_size,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                method='sentence'
            ),
        ]

       

    def substitute(self, data, rs=None):
        if not data:
            return data

        if isinstance(data, list):
            all_data = data
        else:
            if data.strip() == '':
                return data
            all_data = [data]

        return rs['model'].predict(all_data)

if __name__ == '__main__':
    text = 'it is easy to say something but hard to do'
    random_transform = AbstSummSentTransform()
    tran = random_transform(text=text,force_apply=True,n=1)
    print(text)
    print(tran['text']) 

