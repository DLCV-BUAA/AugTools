import warnings
import random
from itertools import product

import cv2
import numpy as np
import os
import re
from collections import defaultdict

from augtools.utils.file_utils import *
from augtools.text.transforms.word.word_transform import WordTransform


class ReservedTransform(WordTransform):
    """
    Augmenter that apply target word replacement for augmentation.
    Can also be used to generate all possible combinations.
    :param float aug_p: Percentage of word will be augmented. 
    :param int aug_min: Minimum number of word will be augmented.
    :param int aug_max: Maximum number of word will be augmented. If None is passed, number of augmentation is
        calculated via aup_p. If calculated result from aug_p is smaller than aug_max, will use calculated result from
        aug_p. Otherwise, using aug_max.
    :param list reserved_tokens: A list of swappable tokens (a list of list). For example, "FWD", "Fwd" and "FW" 
        are referring to "foward" in email communcation while "Sincerely" and "Best Regards" treated as same 
        meaning. The input should be [["FWD", "Fwd", "FW"], ["Sincerely", "Best Regards"]]. 
    :param bool case_sensitive: Default is True. If True, it will only replace alternative token if all cases are same.
    :param bool generate_all_combinations: Default is False. If True, all the possible combinations of sentences
        possible with reserved_tokens will be returned. 
    :param func tokenizer: Customize tokenization process
    :param func reverse_tokenizer: Customize reverse of tokenization process
    :param str name: Name of this augmenter
    """

    def __init__(self, reserved_tokens, case_sensitive=True, generate_all_combinations=False,
                 aug_min=1, aug_max=10, aug_p=0.3, stopwords=None, action='substitute',
                 tokenizer=None, reverse_tokenizer=None, stopwords_regex=None,):
        super().__init__(
            action=action, aug_p=aug_p, aug_min=aug_min, aug_max=aug_max, stopwords=stopwords,
            tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, stopwords_regex=stopwords_regex)
        self.generate_all_combinations = generate_all_combinations

        self.reserved_tokens = reserved_tokens
        self.reserved_lower_tokens = []
        self.case_sensitive = case_sensitive

        self.reserved_token_dict = {}
        self.reserved_phrase_dict = {}
        self.reserved_phrase_concats = []
        self.reserved_phrase_regexs = []
        self.CONNECT_TOKEN = 'nnnnn'
        
        if not case_sensitive:
            self.reserved_lower_tokens = [t.lower() for tokens in reserved_tokens for t in tokens]

        reserved_phrase_dict_by_len = defaultdict(list)
        for i, tokens in enumerate(reserved_tokens):
            for t in tokens:
                if not case_sensitive:
                    t = t.lower()

                phrase_tokens = self.tokenizer(t)
                if len(phrase_tokens) == 1:
                    # For single word
                    # If duplicates word occurs, pick the last one.
                    self.reserved_token_dict[t] = i
                else:
                    # For phrase
                    reserved_phrase_dict_by_len[len(phrase_tokens)].append((t, phrase_tokens, i))

        for i in sorted(reserved_phrase_dict_by_len.keys(), reverse=True):
            for phrase, phrase_tokens, pos in reserved_phrase_dict_by_len[i]:
                phrase_concat_token = self.CONNECT_TOKEN.join(phrase_tokens)
                phrase_token_regex = re.compile('(' + phrase + ')', re.IGNORECASE)

                self.reserved_phrase_dict[phrase_concat_token] = pos
                self.reserved_phrase_concats.append(phrase_concat_token)
                self.reserved_phrase_regexs.append(phrase_token_regex)


    def _skip_aug(self, token_idxes, tokens, model=None):
        results = []
        for idx in token_idxes:
            t = tokens[idx]
            if not self.case_sensitive:
                t = t.lower()

            if t in self.reserved_token_dict:
                # For single word
                results.append(idx)
            elif t in self.reserved_phrase_dict:
                # For phrase
                results.append(idx)
            
        return results
        
    def generate_combinations(self, data, combination=None):
        if(not data):
            combination = list(combination)
            yield combination
        else:
            data = list(data)
        
            if(not combination):
                combination = []
            else:
                combination = list(combination)

            item = data.pop(0)
            #print(item)
            for choice in item[1]:
                combination.append((item[0],choice))
                yield from self.generate_combinations(data,combination)
                
    def substitute(self, data, rs=None):
        if not data or not data.strip():
            return data
        tokens = self._pre_process(data)
        aug_idxes = self._get_aug_idxes(tokens)
        if aug_idxes is None or len(aug_idxes) == 0:
            return data
        
        data_lower = data.lower()
        
        
        if(self.generate_all_combinations):
            assert self.aug_p == 1, "Augmentation probability has to be 1 to genenerate all combinations. Set aug_p=1 in constructor."

            candidate_token_list = []

            for aug_idx in aug_idxes:
                original_token = tokens[aug_idx]
                if not self.case_sensitive:
                    original_token = original_token.lower()

                if original_token in self.reserved_token_dict:
                    candidate_tokens = []
                    for t in self.reserved_tokens[self.reserved_token_dict[original_token]]:
                        candidate_tokens.append(t)
                elif original_token in self.reserved_phrase_concats:
                    candidate_tokens = []
                    for t in self.reserved_tokens[self.reserved_phrase_dict[original_token]]:
                        candidate_tokens.append(t)
                
                candidate_token_list.append((aug_idx,candidate_tokens))
        
            generated_combinations = []
            #print(candidate_token_list)
            
            for candidate_tokens in self.generate_combinations(candidate_token_list):
                #print(candidate_tokens)
                augmented_tokens = tokens
                for candidate_token in candidate_tokens:
                    aug_idx,new_token = candidate_token
                    if aug_idx == 0:
                        new_token = self._align_capitalization(original_token, new_token)
                    augmented_tokens[aug_idx] = new_token
                
                augmented_text = self.reverse_tokenizer(augmented_tokens)

                same_as_original = False
                if self.case_sensitive:
                    same_as_original = augmented_text == data
                else:
                    same_as_original = augmented_text.lower() == data_lower

                if not same_as_original:
                    generated_combinations.append(augmented_text)
            
            return generated_combinations
            # return sorted(generated_combinations) # not sorting to speed up
            
        else:
            augmented_tokens = tokens
            for aug_idx in aug_idxes:
                original_token = tokens[aug_idx]
                if not self.case_sensitive:
                    original_token = original_token.lower()

                if original_token in self.reserved_token_dict:
                    candidate_tokens = []
                    for t in self.reserved_tokens[self.reserved_token_dict[original_token]]:
                        compare_token = t.lower() if not self.case_sensitive else t
                        if compare_token != original_token:
                            candidate_tokens.append(t)
                elif original_token in self.reserved_phrase_concats:
                    candidate_tokens = []
                    for t in self.reserved_tokens[self.reserved_phrase_dict[original_token]]:
                        compare_token = t.replace(' ', self.CONNECT_TOKEN)
                        compare_token = compare_token.lower() if not self.case_sensitive else compare_token
                        if compare_token != original_token:
                            candidate_tokens.append(t)

                new_token = self.sample(candidate_tokens, 1)[0]
                if aug_idx == 0:
                    new_token = self._align_capitalization(original_token, new_token)

                augmented_tokens[aug_idx] = new_token

        return self._post_process(augmented_tokens)



    def _pre_process(self, data=None):
        for reserved_concat_phrase, reserved_phrase_regex in zip(
            self.reserved_phrase_concats, self.reserved_phrase_regexs):
            data = reserved_phrase_regex.sub(reserved_concat_phrase, data)
        tokens = self.tokenizer(data)

        return tokens
    
    def _post_process(self, augmented_tokens):
        return self.reverse_tokenizer(augmented_tokens)


# TODO: not test

if __name__ == '__main__':
    text = 'i eat an apple and hit someone'
    random_transform = ReservedTransform(reserved_tokens=[['i', 'I'], ['eat', 'have', 'enjoy'], ['an', 'a small', 'a red'], 
                                          ['and', 'then'], ['hit', 'run into', 'bump into'], ['someone', 'a person', 'somebody']], generate_all_combinations=False, aug_p=0.3)
    tran = random_transform(text=text,force_apply=True,n=3)
    print(text)
    print(tran['text'])   