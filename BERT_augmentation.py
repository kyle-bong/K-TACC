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
        random.seed(42)
    def random_masking_replacement(self, sentence, ratio=0.15):
        """Masking random eojeol of the sentence, and recover them using PLM.

        Args:
            sentence (str): Source sentence
            ratio (int): Ratio of masking

        Returns:
          str: Recovered sentence
        """
        
        span = int(round(len(sentence.split()) * ratio))
        
        # 품질 유지를 위해, 문장의 어절 수가 4 이하라면 원문장을 그대로 리턴합니다.
        if len(sentence.split()) <= 4:
            return sentence

        mask = self.tokenizer.mask_token
        unmasker = self.unmasker

        unmask_sentence = sentence
        # 처음과 끝 부분을 [MASK]로 치환 후 추론할 때의 품질이 좋지 않음.
        random_idx = random.randint(1, len(unmask_sentence.split()) - span)
        
        unmask_sentence = unmask_sentence.split()
        # del unmask_sentence[random_idx:random_idx+span]
        cache = []
        for _ in range(span):
            # 처음과 끝 부분을 [MASK]로 치환 후 추론할 때의 품질이 좋지 않음.
            while cache and random_idx in cache:
                random_idx = random.randint(1, len(unmask_sentence) - 2)
            cache.append(random_idx)
            unmask_sentence[random_idx] = mask
            unmask_sentence = unmasker(" ".join(unmask_sentence))[0]['sequence']
            unmask_sentence = unmask_sentence.split()
        unmask_sentence = " ".join(unmask_sentence)
        unmask_sentence = unmask_sentence.replace("  ", " ")

        return unmask_sentence.strip()

    def random_masking_insertion(self, sentence, ratio=0.15):
        
        span = int(round(len(sentence.split()) * ratio))
        mask = self.tokenizer.mask_token
        unmasker = self.unmasker
        
        # Recover
        unmask_sentence = sentence
        
        for _ in range(span):
            unmask_sentence = unmask_sentence.split()
            random_idx = random.randint(0, len(unmask_sentence)-1)
            unmask_sentence.insert(random_idx, mask)
            unmask_sentence = unmasker(" ".join(unmask_sentence))[0]['sequence']

        unmask_sentence = unmask_sentence.replace("  ", " ")

        return unmask_sentence.strip()