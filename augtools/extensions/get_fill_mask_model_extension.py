try:
    import torch
    from torch.utils import data as t_data
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from transformers import pipeline
except ImportError:
    # No installation required if not using this function
    pass
from augtools.extensions.extension import Extension
from augtools.utils.file_utils import *
from augtools.utils.text_model_utils import LanguageModels


class GetFMModelExtension(Extension):
    def __init__(self,
                model_path='bert-base-uncased', 
                model_type='bert', 
                top_k=5, 
                device='cuda', 
                max_length=300,
                batch_size=32,
        method='WORD'):
        
        
        self.model = FMModels(
            model_path=model_path, 
            model_type=model_type, 
            top_k=top_k, 
            device=device, 
            max_length=max_length,
            batch_size=batch_size,
        method=method)
        
    def _get_rs(self, rs, **kwargs):
        rs['model'] = self.model
        return rs


class FMModels(LanguageModels):
    def __init__(
        self, 
        model_path='bert-base-uncased', 
        model_type='bert', 
        top_k=5, 
        device='cuda', 
        max_length=300,
        batch_size=32,
        method='word'
    ):
        super().__init__(device=device, model_type=model_type, method=method)
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from transformers import pipeline
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')

        self.max_length = max_length
        self.batch_size = batch_size
        self.model_path = model_path
        device = self.convert_device(device)
        top_k = top_k if top_k else 5
        self.model = pipeline("fill-mask", model=model_path, device=device, top_k=top_k)


    def _predict_word(self, texts, target_words=None, n=1):
        return self._predict(texts, target_words, n)
    
    def _predict_sentence(self, texts, target_words=None, n=1):
        return self._predict(texts, target_words, n)


    def to(self, device):
        self.model.model.to(device)

    def get_device(self):
        return str(self.model.device)

    def get_tokenizer(self):
        return self.model.tokenizer
        

    def get_model(self):
        return self.model.model

    def get_max_num_token(self):
        return self.model.model.config.max_position_embeddings - 2 * 5

    def is_skip_candidate(self, candidate):
        return candidate.startswith(self.get_subword_prefix())

    def token2id(self, token):
        # Iseue 181: TokenizerFast have convert_tokens_to_ids but not convert_tokens_to_id
        if 'TokenizerFast' in self.tokenizer.__class__.__name__:
            # New transformers API
            return self.model.tokenizer.convert_tokens_to_ids(token)
        else:
            # Old transformers API
            return self.model.tokenizer._convert_token_to_id(token)

    def id2token(self, _id):
        return self.model.tokenizer._convert_id_to_token(_id)

    def _predict(self, texts, target_words=None, n=1):
        results = []

        predict_results = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                predict_result = self.model(texts[i:i+self.batch_size])
                if isinstance(predict_result, list) and len(predict_result) > 0:
                    if isinstance(predict_result[0], list):
                        predict_results.extend(predict_result)
                    else:
                        predict_results.extend([predict_result])

        for result in predict_results:
            temp_results = []
            for r in result:
                token = r['token_str']
                if self.model_type in ['bert'] and token.startswith('##'):
                    continue
                # subword came without space for roberta but not normal subowrd prefix
                if self.model_type in ['roberta', 'bart'] and not token.startswith(' '):
                    continue

                temp_results.append(token)

            results.append(temp_results)
    
        return results

    
if __name__ == "__main__":
    
    # extension = GetWordMTModelExtension()
    # rs = None
    # rs = extension(rs)
    # print(rs['model'].predict('you are a bad guy'))
    model = FMModels()
    print(model('{}'.format(model.get_mask_token())))