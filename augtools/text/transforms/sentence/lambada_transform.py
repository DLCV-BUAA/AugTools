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


class LambadaSentTransform(SentenceTransform):

    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_dir: Directory of model. It is generated from train_lambada.sh under scritps folders.n
    :param float threshold: The threshold of classification probabilty for accpeting generated text. Return all result if threshold
        is None.
    :param int batch_size: Batch size.
    :param int min_length: The min length of output text.
    :param int max_length: The max length of output text.
    :param float temperature: The value used to module the next token probabilities.
    :param int top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
    :param float top_p: If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or
        higher are kept for generation.
    :param float repetition_penalty : The parameter for repetition penalty. 1.0 means no penalty.
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. 
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param str name: Name of this augmenter

    """

    def __init__(self, cls_model_dir, gen_model_dir, threshold=0.7, device='cpu', action='insert',repetition_penalty=1.0,
        min_length=100, max_length=300, batch_size=16, temperature=1.0, top_k=50, top_p=0.9,):
        super().__init__(
            action=action, device=device)
        self.aug_src = 'lambada'

        self.min_length = min_length
        self.max_length = max_length
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        self.repetition_penalty = repetition_penalty
        self.threshold = threshold
        self.cls_model_dir = cls_model_dir
        self.gen_model_dir = gen_model_dir
        self.threshold = threshold

        with open(os.path.join(cls_model_dir, 'label_encoder.json')) as json_file:
            self.label2id = json.load(json_file)       

    def _append_extensions(self):
        return [
            GetSentenceModelExtension(name=self.aug_src,
                cls_model_dir=self.cls_model_dir, 
                gen_model_dir = self.gen_model_dir,
                threshold=self.threshold, 
                min_length=self.min_length, 
                device=self.device, 
                max_length=self.max_length,
                batch_size=self.batch_size,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                method='sentence'
            ),
        ]

       

    def insert(self, data, rs=None):
        if not data:
            return data

        if isinstance(data, list):
            all_data = data
        else:
            if data.strip() == '':
                return data
            all_data = [data]

        for d in all_data:
            if d not in self.label2id:
                raise Exception('Label {} does not exist. Possible labels are {}'.format(d, self.label2id.keys()))

        return rs['model'].predict(all_data)
    
if __name__ == '__main__':
    text = ['0', '1' , '2']
    random_transform = LambadaSentTransform(cls_model_dir='augtools/extensions/model/lambada/cls', gen_model_dir='augtools/extensions/model/lambada/gen', threshold=0.3)
    tran = random_transform(text=text, device='cpu', n=5)
    print(text)
    print(tran['text']) 

