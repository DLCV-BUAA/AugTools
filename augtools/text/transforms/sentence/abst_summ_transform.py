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

    def __init__(self, model_path='dongxq/test_model', tokenizer_path='dongxq/test_model', device='cuda', action='substitute',
        min_length=10, max_length=20, batch_size=32, temperature=1.0, top_k=50, top_p=0.9,):
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
    '''
    text = 'On the app itself, perplexed academic followers twittered among themselves as to \
            which of its competitors might offer a substitute platform: Bluesky, perhaps, \
            though it is still in development and you have to be invited to use it, \
            or Mastodon, an early competitor, but which has fewer than 3 million, active users. \
            Twitter does still have some cultural heft: last Tuesday the singer Labi Siffre used it\
            to complain of a series of racist slights he had suffered during a visit to London.\
            But in business terms Mr Musk’s rivals are circling. Just a day after the rebrand news,\
            the video‑sharing platform TikTok responded with its own announcement \
            that it was expanding into text‑only posts. '
    '''
    text = '相关争议事件中，相比于运营企业，消费者往往处于弱势地位，在维权过程中需要与运营企业多次交涉沟通，耗时费力才能挽回损失，有时遭遇运营企业拖延、推诿，甚至陷入维权僵局。此外，一些消费者还可能存在未能及时察觉误扣费或隐私泄露等问题，导致权益受损而不自知。舆论认为，解决共享充电宝行业问题多发的现状，需要各方参与、共管共治。企业在明确标注价格等事项外，还需明示消费者申诉反馈渠道，提高消费纠纷解决效率。'
    random_transform = AbstSummSentTransform()
    tran = random_transform(text=text,force_apply=True,n=1)
    #print(text)
    print(tran['text']) 

