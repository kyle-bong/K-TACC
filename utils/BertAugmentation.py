import transformers
import re
import random
import numpy as np


class BertAugmentation():
    def __init__(self):
        self.model_name = 'monologg/koelectra-base-v3-generator'
        self.model = transformers.AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        self.unmasker = transformers.pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)
        random.seed(42)
        
    def random_masking_replacement(self, sentence: str, ratio: float = 0.15) -> str:
        """어절을 무작위로 마스킹한 후 PLM을 이용해 복원합니다.

        Args:
            sentence (str): Source sentence
            ratio (int): Ratio of masking

        Returns:
          str: Recovered sentence
        """
        
        words = sentence.split()
        num_words = len(words)
        
        # 품질 유지를 위해, 문장의 어절 수가 4 이하라면 원문장을 그대로 리턴합니다.
        if num_words <= 4:
            return sentence

        num_to_mask = max(1, int(round(num_words * ratio))) # 최소 1개의 단어는 무조건 마스킹합니다.
        mask_token = self.tokenizer.mask_token

        # 처음과 끝 부분을 [MASK]로 변환 후 복원하는 것은 품질이 좋지 않아, 처음과 끝 부분은 마스킹에서 제외합니다.
        mask_indices = random.sample(range(1, num_words - 1), num_to_mask)
        
        for idx in mask_indices:
            if idx >= len(words):
                continue
            
            words[idx] = mask_token
            unmasked_sentence = " ".join(words)
            unmasked_sentence = self.unmasker(unmasked_sentence)[0]['sequence']
            words = unmasked_sentence.split()

        return " ".join(words).replace("  ", " ").strip()

    def random_masking_insertion(self, sentence, ratio=0.15):
        """
        문장 내 무작위 위치에 마스크 토큰을 삽입 후 PLM을 이용해 복원합니다. 

        Args:
            sentence (str): Source sentence.
            ratio (float): Proportion of words to mask.

        Returns:
            str: Sentence with inserted mask tokens replaced by model predictions.
        """
        
        words = sentence.split()
        num_words = len(words)
        num_to_insert = max(1, int(round(num_words * ratio)))  
        
        mask_token = self.tokenizer.mask_token
        
        for _ in range(num_to_insert):
            insert_idx = random.randint(0, num_words)
            words.insert(insert_idx, mask_token)
            unmasked_sentence = " ".join(words)
            unmasked_sentence = self.unmasker(unmasked_sentence)[0]['sequence']
            words = unmasked_sentence.split()
        
        return " ".join(words).replace("  ", " ").strip()