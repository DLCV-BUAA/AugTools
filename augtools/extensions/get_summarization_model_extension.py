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


class GetSummarizationModelExtension(Extension):
    def __init__(self,
        model_name='t5-base', 
        tokenizer_name=None, 
        min_length=10, 
        device='cuda', 
        max_length=20,
        batch_size=32,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        method='sentence'):
        
        
        self.model = Summarization(
            model_name=model_name, 
            tokenizer_name=tokenizer_name, 
            min_length=min_length, 
            device=device, 
            max_length=max_length,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            method=method)
            
    def _get_rs(self, rs, **kwargs):
        rs['model'] = self.model
        return rs


class Summarization(LanguageModels):
    def __init__(
        self, 
        model_name='t5-base', 
        tokenizer_name=None, 
        min_length=10, 
        device='cuda', 
        max_length=20,
        batch_size=32,
        temperature=1.0,
        top_k=50,
        top_p=0.9,
        method='sentence'
    ):
        super().__init__(device=device, model_type=None, method=method)
        
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from transformers import pipeline
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Missed transformers library. Install transfomers by `pip install transformers`')

        self.model_name = model_name
        self.tokenizer_name = model_name if tokenizer_name is None else tokenizer_name
        self.min_length = min_length
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.batch_size = batch_size


        device = self.convert_device(device)
        top_k = top_k if top_k else 5
        self.model = pipeline("summarization", model=self.model_name, tokenizer=self.tokenizer_name, 
                device=device, framework="pt")


    def _predict_word(self, texts, target_words=None, n=1):
        return self._predict(texts, target_words, n)
    
    def _predict_sentence(self, texts, target_words=None, n=1):
        return self._predict(texts, target_words, n)


    def get_device(self):
        return str(self.model.device)


    def _predict(self, texts, target_words=None, n=1):
        results = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                predict_result = self.model(texts[i:i+self.batch_size], 
                    min_length=self.min_length, 
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    num_workers=1)
                if isinstance(predict_result, list):
                    results.extend(predict_result)
                else:
                    results.append(predict_result)
        results = [r['summary_text'] for r in results]
        
        return results
        
   
if __name__ == "__main__":
    

    model = Summarization()
    print(model(['it is easy to say something but hard to do']))




