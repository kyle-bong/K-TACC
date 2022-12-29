import transformers
import re
import random

class BERT_Augmentation():
    def __init__(self):
        self.electra_model_name = 'monologg/koelectra-base-v3-generator'
        self.electra_model = transformers.AutoModelForMaskedLM.from_pretrained(self.electra_model_name)
        self.electra_tokenizer = transformers.AutoTokenizer.from_pretrained(self.electra_model_name)
        self.electra_unmasker = transformers.pipeline("fill-mask", model=self.electra_model, tokenizer=self.electra_tokenizer)
        
    def random_masking_replacement(self, sentence, span=1):
        """Masking random eojeol of the sentence, and recover it using PLM.

        Args:
            sentence (_type_): _description_
            mask_token (_type_): _description_
            unmasker (_type_): _description_
            span
        """
        assert span < len(sentence)

        mask = self.electra_tokenizer.mask_token
        unmasker = self.electra_unmasker

        unmask_sentence = sentence.split()
        random_idx = random.randint(0, len(unmask_sentence)-1 - (span-1))
        unmask_sentence[random_idx] = mask
        # 원 문장에서 span 길이만큼 삭제. (span 길이만큼 masking하기 위함.)
        del unmask_sentence[random_idx+1:random_idx+1+span]
        # print('unmask_sentnece: ', unmask_sentence)
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


    def random_masking_insertion(self, sentence):
        mask = self.electra_tokenizer.mask_token
        unmasker = self.electra_unmasker
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

# BERT_aug = BERT_Augmentation()

# func = BERT_aug.random_masking_replacement

# ins = BERT_aug.random_masking_insertion

# sentence = "이순신은 조선 중기의 무신이라고 전해진다."
# for _ in range(5):
#     result = func(sentence, span=2)
#     print(result)


# for _ in range(5):
#     result = ins(sentence)
#     print(result)
