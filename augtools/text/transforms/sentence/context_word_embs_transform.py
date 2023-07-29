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


class ContextualSentTransform(SentenceTransform):

    # https://arxiv.org/pdf/1707.07328.pdf, https://arxiv.org/pdf/2003.02245.pdf
    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_path: Model name or model path. It used transformers to load the model. Tested
        'gpt2', 'distilgpt2'. 
    :param str model_type: Type of model. For XLNet model, use 'xlnet'. For GPT2 or distilgpt2 model, use 'gpt'. If 
        no value is provided, will determine from model name.
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
    :param bool silence: Default is True. transformers library will print out warning message when leveraing
        pre-trained model. Set True to disable the expected warning message.
    :param str name: Name of this augmenter

    """

    def __init__(self, model_path='gpt2', device='cuda', action='insert',
        min_length=100, max_length=500, batch_size=32, temperature=1.0, top_k=50, top_p=0.9,):
        super().__init__(
            action=action, device=device)
        self.aug_src = 'generation'
        self.model_path = model_path
        self.min_length = min_length
        self.max_length = max_length
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        

    def _append_extensions(self):
        return [
            GetSentenceModelExtension(name=self.aug_src,
                model_path=self.model_path, 
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

       

    def insert(self, data, rs=None):
        if not data:
            return data

        if isinstance(data, str):
            if data.strip() == '':
                return data
            all_data = [data]
        elif isinstance(data, Iterable):
            all_data = data
        else:
            all_data = [data]

        return rs['model'].predict(all_data)

if __name__ == '__main__':
    text = 'it is easy to say something but hard to do'
    random_transform = ContextualSentTransform()
    tran = random_transform(text=text,force_apply=True,n=1)
    print(text)
    print(tran['text']) 

