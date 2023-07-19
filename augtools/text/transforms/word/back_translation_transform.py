import warnings
import random
from itertools import product

import cv2
import numpy as np

from augtools.text.transforms.word.word_transform import WordTransform
from augtools.extensions.get_word_dict_model_extension import GetWordDcitModelExtension


class BackTranslationTransform(WordTransform):
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

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.BackTranslationAug()
    """

    def __init__(self, from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en',
        device='cuda', batch_size=32, max_length=300, action='subtitute', silence=True):
        super().__init__(
            action=action, aug_p=None, aug_min=None, aug_max=None, tokenizer=None, reverse_tokenizer=None)
        self.aug_src = 'machine_translation'
        
        self.from_model_name = from_model_name
        self.to_model_name = to_model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.silence = silence

    def _append_extensions(self):
        
        return [
            GetWordDcitModelExtension(name=self.aug_src, 
                                      
                src_model_name=self.from_model_name,
                tgt_model_name=self.to_model_name, 
                device=self.device,
                batch_size=self.batch_size,
                max_length=self.max_length,
                silence=self.silence,
                method='word'),
        ]
    
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
            new_token = rs['model'].predict(original_token)

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
    backtrans_transform = BackTranslationTransform(action='subtitute')
    tran = backtrans_transform(text=text,force_apply=True,n=3)
    print(text)
    print(tran['text'])  