"""
    Augmenter that apply operation (word level) to textual input based on contextual word embeddings.
"""

import warnings
import random
from itertools import product
import string

import cv2
import numpy as np
import re

from augtools.text.transforms.word.word_transform import WordTransform
from augtools.extensions.get_word_dict_model_extension import GetWordDcitModelExtension


class ContextualWordEmbsTransform(WordTransform):
    # https://arxiv.org/pdf/1805.06201.pdf, https://arxiv.org/pdf/2003.02245.pdf
    """
    Augmenter that leverage contextual word embeddings to find top n similar word for augmentation.

    :param str model_path: Model name or model path. It used transformers to load the model. Tested
        'bert-base-uncased', 'bert-base-cased', 'distilbert-base-uncased', 'roberta-base', 'distilroberta-base',
        'facebook/bart-base', 'squeezebert/squeezebert-uncased'.
    :param str model_type: Type of model. For BERT model, use 'bert'. For RoBERTa/LongFormer model, use 'roberta'. 
        For BART model, use 'bart'. If no value is provided, will determine from model name.
    :param str action: Either 'insert or 'substitute'. If value is 'insert', a new word will be injected to random
        position according to contextual word embeddings calculation. If value is 'substitute', word will be replaced
        according to contextual embeddings calculation
    :param int top_k: Controlling lucky draw pool. Top k score token will be used for augmentation. Larger k, more
        token can be used. Default value is 100. If value is None which means using all possible tokens.
    :param float aug_p: Percentage of word will be augmented.
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list stopwords: List of words which will be skipped from augment operation. Do NOT include the UNKNOWN word.
        UNKNOWN word of BERT is [UNK]. UNKNOWN word of RoBERTa and BART is <unk>.
    :param str stopwords_regex: Regular expression for matching words which will be skipped from augment operation.
    :param str device: Default value is CPU. If value is CPU, it uses CPU for processing. If value is CUDA, it uses GPU
        for processing. Possible values include 'cuda' and 'cpu'. (May able to use other options)
    :param int batch_size: Batch size.
    :param bool force_reload: Force reload the contextual word embeddings model to memory when initialize the class.
        Default value is False and suggesting to keep it as False if performance is the consideration.
    :param bool silence: Default is True. transformers library will print out warning message when leveraing
        pre-trained model. Set True to disable the expected warning message.
    :param str name: Name of this augmenter
    :param language: Indicate the language the augmenting text belongs to
    :param normal: English LM all obey the normal rules defined in class LanguageModels, but Chinese LM and Multi-Language LM usually 
        don't follow the rules, when not, param prefix and param all are supposed to be set manually
    :param prefix: prefix the model employs
    :param mask_token: mask_token the model employs
    :param all: if the prefix is applied to all tokens, set param all True; elif the prefix is only added to the first token,
        set param all False

    >>> import nlpaug.augmenter.word as naw
    >>> aug = naw.ContextualWordEmbsAug()
    """

    def __init__(self, model_path='roberta-base', model_type='roberta', action="substitute", top_k=20, 
                 aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                 batch_size=32, device='cpu', stopwords_regex=None, language="en", normal=True, prefix=None, mask_token=None, all=True):
        super().__init__(
            action=action, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, tokenizer=None,
            stopwords=stopwords, stopwords_regex=stopwords_regex,)
        self.aug_src = 'fill_mask_model'
        
        self.model_path = model_path
        self.model_type = model_type if model_type != '' else self.check_model_type() 
        self.top_k = top_k
        self.device = device
        self.batch_size = batch_size

        if stopwords and 'uncased' in model_path:
            self.stopwords = [s.lower() for s in self.stopwords]

        self.stopword_reg = None
        self.reserve_word_reg = None
        self._build_stop_words(stopwords)
        self.language = language
        self.normal = normal
        self.prefix = prefix
        self.mask_token = mask_token
        self.all = all


    def _build_stop_words(self, stopwords, model=None):
        if stopwords:
            prefix_reg = '(?<=\s|\W)'
            suffix_reg = '(?=\s|\W)'
            stopword_reg = '('+')|('.join([prefix_reg + re.escape(s) + suffix_reg for s in stopwords])+')'
            self.stopword_reg = re.compile(stopword_reg)
            unknown_token = model.get_unknown_token() or model.UNKNOWN_TOKEN
            reserve_word_reg = '(' + prefix_reg + re.escape(unknown_token) + suffix_reg + ')'
            self.reserve_word_reg = re.compile(reserve_word_reg)
    
    def _append_extensions(self):
        
        return [
            GetWordDcitModelExtension(name=self.aug_src,                           
                model_path=self.model_path, 
                model_type=self.model_type, 
                top_k=self.top_k, 
                device=self.device, 
                batch_size=self.batch_size ,
                method='word',
                normal=self.normal,
                prefix=self.prefix,
                ),
        ]

    def check_model_type(self):
        # if 'xlnet' in self.model_path.lower():
        #     return 'xlnet'

        if 'longformer' in self.model_path.lower():
            return 'roberta' 
        elif 'roberta' in self.model_path.lower():
            return 'roberta'

        elif 'distilbert' in self.model_path.lower():
            return 'bert'
        elif 'squeezebert' in self.model_path.lower():
            return 'bert'
        elif 'bert' in self.model_path.lower():
            return 'bert'

        elif 'bart' in self.model_path.lower():
            return 'bart'
        

#     'google/electra-small-discriminator',
#     'google/reformer-enwik8',
#     'funnel-transformer/small-base',
#     'google/tapas-base',
#     'microsoft/deberta-base'
        
        return ''

    def is_stop_words(self, token, model=None):
        # Will execute before any tokenization. No need to handle prefix processing
        if self.stopwords:
            unknown_token = model.get_unknown_token() or model.UNKNOWN_TOKEN
            if token == unknown_token:
                return True
            return token.lower() in self.stopwords
        else:
            return False

    def _skip_aug(self, token_idxes, tokens, model=None):
        results = []
        #print(tokens)
        if self.normal:
            for token_idx in token_idxes:
                token = tokens[token_idx]

                # Do not augment subword
                if self.model_type in ['bert', 'electra'] \
                    and token.startswith(model.get_subword_prefix()):
                    continue
                # Do not augment tokens if len is less than aug_min
                if (model.get_subword_prefix() in token and len(token) < self.aug_min+1) \
                    or (model.get_subword_prefix() not in token and len(token) < self.aug_min):
                    continue
                if self.model_type in ['xlnet', 'roberta', 'bart']:
                    # xlent may tokenize word incorrectly. For example, 'fox', will be tokeinzed as ['_', 'fox']
                    if token == model.get_subword_prefix():
                        continue

                    # subword
                    if not token.startswith(model.get_subword_prefix()):
                        continue

                results.append(token_idx)
        else:
            if self.all:
                for token_idx in token_idxes:
                    token = tokens[token_idx]

                    if token == self.prefix:
                        continue

                    if not token.startswith(self.prefix):
                        continue
                    results.append(token_idx)
            else:
                return token_idxes

        return results

    def split_text(self, data, model=None):
       
        if self.stopwords:
            unknown_token = model.get_unknown_token() or model.UNKNOWN_TOKEN
            preprocessed_data, reserved_stopwords = self.replace_stopword_by_reserved_word(data, self.stopword_reg, unknown_token)
        else:
            preprocessed_data, reserved_stopwords = data, None
        tokens = model.get_tokenizer().tokenize(preprocessed_data)
        #print(tokens)

        if model.get_model().config.max_position_embeddings == -1:  # e.g. No max length restriction for XLNet
            return (preprocessed_data, None, tokens, None), reserved_stopwords  # (Head text, tail text, head token, tail token), reserved_stopwords

        ids = model.get_tokenizer().convert_tokens_to_ids(tokens[:self.max_num_token])
        head_text = model.get_tokenizer().decode(ids).strip()
        # head_text = model.get_tokenizer().convert_tokens_to_string(tokens[:self.max_num_token]).strip()
        tail_text = None
        if len(tokens) >= self.max_num_token:
            # tail_text = model.get_tokenizer().convert_tokens_to_string(tokens[self.max_num_token:]).strip()
            ids = model.get_tokenizer().convert_tokens_to_ids(tokens[self.max_num_token:])
            tail_text = model.get_tokenizer().decode(ids).strip()

        #print(head_text)
        return (head_text, tail_text, tokens[:self.max_num_token], tokens[self.max_num_token:]), reserved_stopwords

    def insert(self, data, rs=None):
        if not data or not data.strip():
            return data
        self.max_num_token = rs['model'].get_max_num_token()
        
        split_results, reserved_stopwords = self._pre_process(data, rs['model'])
        #print(split_results[0])

        if self.normal:
            prefix = rs['model'].get_subword_prefix()
        else:
            prefix = self.prefix
        
        for i, (split_result, reserved_stopword_tokens) in enumerate(zip(split_results, reserved_stopwords)):
            head_text, tail_text, head_tokens, tail_tokens = split_result
            #print(split_result)            
            if self.model_type in ['xlnet', 'roberta', 'bart']:
                # xlent and roberta tokens include prefix (e.g. ▁ or Ġ')
                cleaned_head_tokens = [t.replace(prefix, '') for t in head_tokens]
                #print(cleaned_head_tokens)
            else:
                cleaned_head_tokens = head_tokens

            # generate aug_idx and subsitute mask
            aug_idxes = self._get_aug_idxes(head_tokens, rs['model'])
            aug_idxes.sort(reverse=True)
            #print(split_results[0])

            if reserved_stopword_tokens:
                cleaned_head_tokens = self.substitute_back_reserved_stopwords(
                    cleaned_head_tokens, reserved_stopword_tokens, rs['model'])

            split_results[i] += (cleaned_head_tokens, aug_idxes, )
            #print(split_results[0])

        # Pad aug_idxes
        max_aug_size = max([len(split_result[5]) for split_result in split_results])
        #print(split_results)
    
        for split_result in split_results:
            aug_idxes = split_result[5]
            for _ in range(max_aug_size - len(aug_idxes)):
                aug_idxes.append(-1)

        if self.mask_token:
            token_placeholder = self.mask_token
        else:
            token_placeholder = rs['model'].get_mask_token()
        
        
        '''
        if self.model_type in ['xlnet', 'roberta', 'bart']:
            token_placeholder = rs['model'].get_subword_prefix() + token_placeholder  # Adding prefix for
            #print(token_placeholder)
        '''
        #print(split_results[0][4])

        # Augment same index of aug by batch
        for i in range(max_aug_size):
            masked_texts = []
            aug_input_poses = [] # store which input augmented. No record if padding

            for j, split_result in enumerate(split_results):
                #print(head_tokens)
                head_tokens, aug_idx = split_result[4], split_result[5][i]
                # -1 if it is padding 
                if aug_idx == -1:
                    continue

                head_tokens.insert(aug_idx, token_placeholder)
                aug_input_poses.append(j)
                #print(head_tokens)

                #print(head_tokens)
                # some tokenizers handle special charas (e.g. don't can merge after decode)
                '''
                if self.model_type in ['bert', 'electra']:
                    ids = rs['model'].get_tokenizer().convert_tokens_to_ids(head_tokens)
                    masked_text = rs['model'].get_tokenizer().decode(ids).strip()
                elif self.model_type in ['xlnet', 'roberta', 'bart']:
                    masked_text = rs['model'].get_tokenizer().convert_tokens_to_string(head_tokens).strip()
                '''
                if self.language in ["cn"]:
                    masked_text = "".join(head_tokens)
                else:
                    masked_text = " ".join(head_tokens)
                masked_texts.append(masked_text)
                #print(masked_texts)

            if not len(masked_texts):
                continue

            outputs = rs['model'].predict(masked_texts)
            #print(outputs)

            for aug_input_pos, output, masked_text in zip(aug_input_poses, outputs, masked_texts):
                split_result = split_results[aug_input_pos]
                head_tokens = split_result[4]
                aug_idx = split_result[5][i] # augment position in text

                # TODO: Alternative method better than dropout
                candidate = ''
                if len(output) == 0:
                    # TODO: no result?
                    pass
                elif len(output) == 1:
                    candidate = output[0]
                elif len(output) > 1:
                    candidate = self.sample(output, 1)[0]

                if candidate == '':
                    continue

                candidate = candidate.strip()
                head_tokens[aug_idx] = candidate

                if len(head_tokens) > self.max_num_token:
                    for j in range(i+1, max_aug_size):
                        split_results[aug_input_pos][5][j] = -1


        augmented_texts = []
       
        for split_result in split_results:
            tail_text, head_tokens = split_result[1], split_result[4]
            #print(head_tokens)
            '''
            ids = rs['model'].get_tokenizer().convert_tokens_to_ids(head_tokens)
            #print(ids)
            if self.model_type=="bert":
                augmented_text = rs['model'].get_tokenizer().decode(ids)
            else:
                augmented_text=""
                for id in ids:
                    augmented_text += rs['model'].get_tokenizer().decode(id) + " "
                augmented_text = augmented_text.rstrip()
            #print(augmented_text)
            '''
            if self.language in ["cn"]:
                augmented_text = "".join(head_tokens)
            else:
                augmented_text = " ".join(head_tokens)

            if tail_text is not None:
                augmented_text += ' ' + tail_text
            augmented_texts.append(augmented_text)

    
        if isinstance(data, list):
            return augmented_texts
        else:
            return augmented_texts[0]

    

    def substitute(self, data, rs=None):
        """
        1. get masked text and mask idx
        2. get subsitute tokens and sample 1
        3. generate result text 
        """
        if not data or not data.strip():
            return data
        
        #print(data)
        if self.normal:
            prefix = rs['model'].get_subword_prefix()
        else:
            prefix = self.prefix

        #print(prefix)
        self.max_num_token = rs['model'].get_max_num_token()
        #print(self.max_num_token)
        
        split_results, reserved_stopwords = self._pre_process(data, rs['model'])
        #print(split_results)
        
        for i, (split_result, reserved_stopword_tokens) in enumerate(zip(split_results, reserved_stopwords)):
            #print(split_result)
            head_text, tail_text, head_tokens, tail_tokens = split_result            
            if self.model_type in ['xlnet', 'roberta', 'bart']:
                # xlent and roberta tokens include prefix (e.g. ▁ or Ġ')
                cleaned_head_tokens = [t.replace(prefix, '') for t in head_tokens]
                #a = '▁今天'.replace('▁', '')
                #print(a)
                #print(cleaned_head_tokens)
            else:
                cleaned_head_tokens = head_tokens

            # generate aug_idx and subsitute mask
            aug_idxes = self._get_aug_idxes(head_tokens, rs['model'])
            aug_idxes.sort(reverse=True)
            
            if reserved_stopword_tokens:
                cleaned_head_tokens = self.substitute_back_reserved_stopwords(
                    cleaned_head_tokens, reserved_stopword_tokens, rs['model'])

            split_results[i] += (cleaned_head_tokens, aug_idxes, )

        # Pad aug_idxes
        max_aug_size = max([len(split_result[5]) for split_result in split_results])
        for split_result in split_results:
            aug_idxes = split_result[5]
            for _ in range(max_aug_size - len(aug_idxes)):
                aug_idxes.append(-1)

        if self.mask_token:
            token_placeholder = self.mask_token
        else:
            token_placeholder = rs['model'].get_mask_token()

        '''
        if self.model_type in ['xlnet', 'roberta', 'bart']:
            token_placeholder = rs['model'].get_subword_prefix() + token_placeholder  # Adding prefix for
        '''
        # Augment same index of aug by batch
        #print(split_results)
        for i in range(max_aug_size):
            original_tokens = []
            masked_texts = []
            aug_input_poses = [] # store which input augmented. No record if padding

            for j, split_result in enumerate(split_results):
                
                head_tokens, aug_idx = split_result[4], split_result[5][i]
                #print(head_tokens)
                if aug_idx == -1:
                    continue
                original_token = head_tokens[aug_idx]
                # -1 if it is padding 

                original_tokens.append(original_token)
                head_tokens[aug_idx] = token_placeholder

                '''
                # remove continuous sub-word
                to_remove_idxes = []
                for k in range(aug_idx+1, len(head_tokens)):
                    subword_token = head_tokens[k]
                    if subword_token in string.punctuation:
                        break
                    if self.model_type in ['bert', 'electra'] and rs['model'].get_subword_prefix() in subword_token:
                        to_remove_idxes.append(k)
                    elif self.model_type in ['xlnet', 'roberta', 'bart'] and rs['model'].get_subword_prefix() not in subword_token:
                        to_remove_idxes.append(k)
                    else:
                        break
                for k in reversed(to_remove_idxes):
                    head_tokens[k] = ''
                '''
                #print(head_tokens)
                aug_input_poses.append(j)

                # some tokenizers handle special charas (e.g. don't can merge after decode)
                '''
                if self.model_type in ['bert', 'electra']:
                    ids = rs['model'].get_tokenizer().convert_tokens_to_ids(head_tokens)
                    masked_text = rs['model'].get_tokenizer().decode(ids).strip()
                elif self.model_type in ['xlnet', 'roberta', 'bart']:
                    masked_text = rs['model'].get_tokenizer().convert_tokens_to_string(head_tokens).strip()
                '''
                #print(head_tokens)
                if self.language in ["cn"]:
                    masked_text = "".join(head_tokens)
                else:
                    masked_text = " ".join(head_tokens)
                #print(masked_text)
                masked_texts.append(masked_text)
                # split_result[4], split_result[5][i] = head_tokens, aug_idx 

            #print(masked_texts)
            if not len(masked_texts):
                continue
            # print(i, head_tokens)
            # print(i, masked_texts, aug_idxes)
            outputs = rs['model'].predict(masked_texts, target_words=original_tokens, n=2)
            #print(outputs)

            # Update doc
            for original_token, aug_input_pos, output, masked_text in zip(original_tokens, aug_input_poses, outputs, masked_texts):
                split_result = split_results[aug_input_pos]
                head_tokens = split_result[4]
                aug_idx = split_result[5][i] # augment position in text

                # TODO: Alternative method better than dropout
                candidate = ''
                if len(output) == 0:
                    # TODO: no result?
                    pass
                elif len(output) == 1:
                    candidate = output[0]
                elif len(output) > 1:
                    candidate = self.sample(output, 1)[0]

                if candidate == '':
                    candidate = original_token

                candidate = candidate.strip()
                head_tokens[aug_idx] = candidate
                #print(candidate)
                #print(head_tokens)

                if len(head_tokens) > self.max_num_token:
                    for j in range(i+1, max_aug_size):
                        split_results[aug_input_pos][5][j] = -1
                
                # split_result[4], split_result[5][i] = head_tokens, aug_idx 

        augmented_texts = []
        for split_result in split_results:
            tail_text, head_tokens = split_result[1], split_result[4]
            #print(head_tokens)
            if self.language in ["cn"]:
                augmented_text = "".join(head_tokens)
            else:
                augmented_text = " ".join(head_tokens)
            #print(head_tokens)
            # 此处的head_tokens拼接起来就是答案了为什么要把token转成id再转回来呢？
            '''
            ids = rs['model'].get_tokenizer().convert_tokens_to_ids(head_tokens)
            augmented_text = rs['model'].get_tokenizer().decode(ids)

            if tail_text is not None:
                augmented_text += ' ' + tail_text

            if self.model_type in ['bert'] :
                augmented_text = rs['model'].get_tokenizer().decode(ids)
            else:
                augmented_text=""
                for id in ids:
                    augmented_text += rs['model'].get_tokenizer().decode(id) + " "
                augmented_text = augmented_text.rstrip()
            '''
            augmented_texts.append(augmented_text)


        if isinstance(data, list):
            return augmented_texts
        else:
            return augmented_texts[0]


    def substitute_back_reserved_stopwords(self, tokens, reserved_stopword_tokens, model):
        unknown_token = model.get_unknown_token() or model.UNKNOWN_TOKEN
        reserved_pos = len(reserved_stopword_tokens) - 1
        for token_i, token in enumerate(tokens):
            if token == unknown_token:
                tokens[token_i] = reserved_stopword_tokens[reserved_pos]
                reserved_pos -= 1
        return tokens
    
    def _pre_process(self, data=None, model=None):
        if not data:
            return data

        if isinstance(data, list):
            all_data = data
        else:
            if data.strip() == '':
                return data

            all_data = [data]

        split_results = [] # head_text, tail_text, head_tokens, tail_tokens
        reserved_stopwords = []
        for d in all_data:
            split_result, reserved_stopword = self.split_text(d, model)
            split_results.append(split_result)
            reserved_stopwords.append(reserved_stopword)
        #print(split_results)
        return split_results, reserved_stopwords


if __name__ == '__main__':
    text = 'If there is no light in the world, I will be the light.'
    #text = '今天天气不错，很适合出去玩。'
    backtrans_transform = ContextualWordEmbsTransform(action='substitute', aug_p=0.1)
    tran = backtrans_transform(text=text,force_apply=True,n=3)
    print(text)
    print(tran['text'])  