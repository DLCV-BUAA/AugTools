import warnings
import random
from itertools import product

import cv2
import numpy as np

from augtools.text.transforms.sentence.sentence_transform import SentenceTransform
from augtools.extensions.get_word_dict_model_extension import GetWordDcitModelExtension


class BackTranslationSentenceTransform(SentenceTransform):
    # https://arxiv.org/pdf/1511.06709.pdf
    """
    Augmenter that leverage two translation models for augmentation. For example, the source is English. This
    augmenter translate source to German and translating it back to English. For detail, you may visit
    https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28

    :param str from_model_name: Any model from https://huggingface.co/models?filter=translation&search=Helsinki-NLP. As
        long as from_model_name is pair with to_model_name. For example, from_model_name is English to Japanese,
        then to_model_name should be Japanese to English.
    :param str to_model_name: Any model from https://huggingface.co/models?filter=translation&search=Helsinki-NLP.
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param int batch_size: Batch size.
    :param int max_length: The max length of output text.
    :param str name: Name of this augmenter

    """

    def __init__(self, from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en',
        device='cuda', batch_size=32, max_length=300, action='substitute'):
        super().__init__(
            action=action, aug_p=None, aug_min=None, aug_max=None, tokenizer=None, reverse_tokenizer=None)
        self.aug_src = 'machine_translation'
        
        self.from_model_name = from_model_name
        self.to_model_name = to_model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

    def _append_extensions(self):
        
        return [
            GetWordDcitModelExtension(name=self.aug_src, 
                                      
                src_model_name=self.from_model_name,
                tgt_model_name=self.to_model_name, 
                device=self.device,
                batch_size=self.batch_size,
                max_length=self.max_length,
                method='sentence'),
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
        augmented_text = rs['model'].predict(all_data)
        return augmented_text
        
        
if __name__ == '__main__':
    text = 'it is easy to say something but hard to do'
    backtrans_transform = BackTranslationSentenceTransform(action='substitute')
    tran = backtrans_transform(text=text,force_apply=True,n=3)
    print(tran['text'])  