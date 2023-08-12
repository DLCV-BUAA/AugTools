from augtools.extensions.extension import Extension
from augtools.utils.file_utils import *
from augtools.utils.text_model_utils import BaseModel
import nltk
from nltk.corpus import wordnet

class GetWordNetModelExtension(Extension):
    def __init__(self, lang, is_synonym=True, method='WORD'):
        self.method = method
        self.model = WordNet(lang, is_synonym, method)
    def _get_rs(self, rs, **kwargs):
        rs['model'] = self.model
        return rs
 

class WordNet(BaseModel):
    def __init__(self, lang, is_synonym=True, method='WORD'):
        super().__init__(method)
        self.model = self._get_model()
        self.lang = lang
        self.is_synonym = is_synonym

    
    def _predict_word(self, data, pos=None):
        results = []
        for synonym in self.model.synsets(data, pos=pos, lang=self.lang):
            for lemma in synonym.lemmas(lang=self.lang):
                #print(lemma)
                if self.is_synonym:
                    results.append(lemma.name())
                else:
                    for antonym in lemma.antonyms():
                        results.append(antonym.name())
        return results
  
    def _get_model(self):
        try:    
            wordnet.synsets('testing')
            return wordnet
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            return wordnet
    
    def pos_tag(self, tokens):
        try:
            results = nltk.pos_tag(tokens)
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
            results = nltk.pos_tag(tokens)

        return results
    
    
if __name__ == "__main__":
    
    extension = GetWordNetModelExtension(lang='eng')
    rs = None
    rs = extension(rs)
    print(rs['model'].predict('eat'))