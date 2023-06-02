import transformers
import re
import random
import numpy as np

class BERT_Augmentation():
    def __init__(self):
        self.model_name = 'monologg/koelectra-base-v3-generator'
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.unmasker = transformers.pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)

    def random_masking_replacement(self, sentence, span_ratio=0.15):
        """Masking random eojeol of the sentence, and recover them using PLM.

        Args:
            sentence (str): Source sentence
            span_ratio (int): Span ratio of masking

        Returns:
          str: Recovered sentence
        """
        
        span = max(1, int(round(len(sentence.split()) * span_ratio)))
        
        # 문장의 어절 수가 1 이하라면 원문장을 그대로 리턴합니다.
        if len(sentence.split()) <= 1:
            return sentence

        mask = self.tokenizer.mask_token
        unmasker = self.unmasker

        unmask_sentence = sentence
        random_idx = random.randint(0, len(unmask_sentence.split())-1 - span)
        
        unmask_sentence = unmask_sentence.split()
        del unmask_sentence[random_idx:random_idx+span]
        unmask_sentence.insert(random_idx, mask)
        # print('unmask_sentence: ', unmask_sentence)
        unmask_sentence = unmasker(" ".join(unmask_sentence))[0]['sequence']

        unmask_sentence = unmask_sentence.replace("  ", " ")

        return unmask_sentence

    def random_masking_insertion(self, sentence, span_ratio=0.15):
        
        span = max(1, int(round(len(sentence.split()) * span_ratio)))
        mask = self.tokenizer.mask_token
        unmasker = self.unmasker
        
        # Recover
        unmask_sentence = sentence
        random_idx = random.randint(-1, len(unmask_sentence.split()))
        
        for _ in range(span):
            unmask_sentence = unmask_sentence.split()
            unmask_sentence.insert(random_idx, mask)
            unmask_sentence = unmasker(" ".join(unmask_sentence))[0]['sequence']

        unmask_sentence = unmask_sentence.replace("  ", " ")

        return unmask_sentence