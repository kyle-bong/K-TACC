"""
class

func 1. mask_replacement

func 2. mask_insertion


"""

import pandas as pd
from ast import literal_eval
import transformers
import re
import random
from collections import defaultdict
import argparse
import time
# from py_hanspell.hanspell import spell_checker


class BERT_Augmentation():
    def __init__(self):
        # self.roberta_model_name = 'klue/roberta-base'
        # self.roberta_model = transformers.AutoModelForMaskedLM.from_pretrained(self.roberta_model_name)
        # self.roberta_tokenizer = transformers.AutoTokenizer.from_pretrained(self.roberta_model_name)
        # self.robert_unmasker = transformers.pipeline("fill-mask", model=self.roberta_model, tokenizer=self.roberta_tokenizer)

        # self.electra_model_name = 'monologg/koelectra-base-v3-generator'
        # self.electra_model = transformers.AutoModelForMaskedLM.from_pretrained(self.electra_model_name)
        # self.electra_tokenizer = transformers.AutoTokenizer.from_pretrained(self.electra_model_name)
        # self.electra_unmasker = transformers.pipeline("fill-mask", model=self.electra_model, tokenizer=self.electra_tokenizer)
        
        self.span = 1

    ## TODO: span 적용하기
    def random_masking_replacement(self, sentence, mask, unmasker, span=1):
        """Masking random eojeol of the sentence, and recover it using PLM.

        Args:
            sentence (_type_): _description_
            mask_token (_type_): _description_
            unmasker (_type_): _description_
            span
        """
        unmask_sentence = sentence.split()
        random_idx = random.randint(0, len(unmask_sentence)-1)
        unmask_sentence[random_idx] = mask
        unmask_result = unmasker(" ".join(unmask_sentence))

        # unmask_token에 '##" 이나 특수기호만 들어가는 것을 방지.
        if not re.findall('\W', unmask_result[0]['token_str']):
            unmask_token = unmask_result[0]['token_str']
            unmask_sentence[random_idx] = unmask_token
            
        elif not re.findall('\W', unmask_result[1]['token_str']):
            unmask_token = unmask_result[1]['token_str']
            unmask_sentence[random_idx] = unmask_token
            
        elif not re.findall('\W', unmask_result[2]['token_str']):
            unmask_token = unmask_result[2]['token_str']
            unmask_sentence[random_idx] = unmask_token
            
        elif not re.findall('\W', unmask_result[3]['token_str']):
            unmask_token = unmask_result[3]['token_str']
            unmask_sentence[random_idx] = unmask_token
            
        elif not re.findall('\W', unmask_result[4]['token_str']):
            unmask_token = unmask_result[4]['token_str']
            unmask_sentence[random_idx] = unmask_token
        else:
            # 만족하는 경우가 없다면 해당 문장에 대해서는 그냥 masking 취소.
            return sentence

        unmask_sentence = " ".join(unmask_sentence) 
        unmask_sentence = unmask_sentence.replace("  ", " ")

        return unmask_sentence


    def random_masking_insertion(self, sentence, mask, unmasker):
        unmask_sentence = sentence.split()
        random_idx = random.randint(0, len(unmask_sentence)-1)
        unmask_sentence.insert(random_idx, mask)

        # Recover
        unmask_result = unmasker(" ".join(unmask_sentence))

        # unmask_token에 '##" 이나 특수기호만 들어가는 것을 방지.
        if not re.findall('\W', unmask_result[0]['token_str']):
            unmask_token = unmask_result[0]['token_str']
            unmask_sentence[random_idx] = unmask_token
            
        elif not re.findall('\W', unmask_result[1]['token_str']):
            unmask_token = unmask_result[1]['token_str']
            unmask_sentence[random_idx] = unmask_token
            
        elif not re.findall('\W', unmask_result[2]['token_str']):
            unmask_token = unmask_result[2]['token_str']
            unmask_sentence[random_idx] = unmask_token
            
        elif not re.findall('\W', unmask_result[3]['token_str']):
            unmask_token = unmask_result[3]['token_str']
            unmask_sentence[random_idx] = unmask_token
            
        elif not re.findall('\W', unmask_result[4]['token_str']):
            unmask_token = unmask_result[4]['token_str']
            unmask_sentence[random_idx] = unmask_token
        else:
            # 만족하는 경우가 없다면 해당 문장에 대해서는 그냥 masking 취소.
            return sentence

        unmask_sentence = " ".join(unmask_sentence) 
        unmask_sentence = unmask_sentence.replace("  ", " ")

        return unmask_sentence


roberta_model_name = 'klue/roberta-base'
roberta_model = transformers.AutoModelForMaskedLM.from_pretrained(roberta_model_name, local_files_only=True)
roberta_tokenizer = transformers.AutoTokenizer.from_pretrained(roberta_model_name)
roberta_unmasker = transformers.pipeline("fill-mask", model=roberta_model, tokenizer=roberta_tokenizer)

electra_model_name = 'monologg/koelectra-base-v3-generator'
electra_model = transformers.AutoModelForMaskedLM.from_pretrained(electra_model_name, local_files_only=True)
electra_tokenizer = transformers.AutoTokenizer.from_pretrained(electra_model_name)
electra_unmasker = transformers.pipeline("fill-mask", model=electra_model, tokenizer=electra_tokenizer)

BERT_aug = BERT_Augmentation()

func = BERT_aug.random_masking_replacement
func2 = BERT_aug.random_masking_replacement

ins = BERT_aug.random_masking_insertion
ins2 = BERT_aug.random_masking_insertion

sentence = "이순신은 조선 중기의 무신이라고 전해진다."
# for _ in range(5):
#     result = func(sentence, roberta_tokenizer.mask_token, roberta_unmasker)
#     print(result)

# for _ in range(5):
#     result = func2(sentence, electra_tokenizer.mask_token, electra_unmasker)
#     print(result)

for _ in range(5):
    result = ins(sentence, roberta_tokenizer.mask_token, roberta_unmasker)
    print(result)

for _ in range(5):
    result = ins2(sentence, electra_tokenizer.mask_token, electra_unmasker)
    print(result)
