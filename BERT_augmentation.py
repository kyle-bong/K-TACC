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
        """Masking random eojeol of the sentence, and recover them using PLM.

        Args:
            sentence (str): Source sentence
            span (int): Span of masking

        Returns:
          str: Masked and recovered sentence
        """
        assert span < len(sentence.split())

        mask = self.electra_tokenizer.mask_token
        unmasker = self.electra_unmasker

        unmask_sentence = sentence
        random_idx = random.randint(0, len(unmask_sentence.split())-1 - span)
        
        unmask_sentence = unmask_sentence.split()
        del unmask_sentence[random_idx:random_idx+span]
        unmask_sentence.insert(random_idx, mask)
        # print('unmask_sentence: ', unmask_sentence)
        unmask_sentence = unmasker(" ".join(unmask_sentence))[0]['sequence']

        unmask_sentence = unmask_sentence.replace("  ", " ")

        return unmask_sentence

    def random_masking_insertion(self, sentence, num=1):
        mask = self.electra_tokenizer.mask_token
        unmasker = self.electra_unmasker

        # Recover
        unmask_sentence = sentence
        random_idx = random.randint(0, len(unmask_sentence.split())-1)
        
        for _ in range(num):
          unmask_sentence = unmask_sentence.split()
          unmask_sentence.insert(random_idx, mask)
          unmask_sentence = unmasker(" ".join(unmask_sentence))[0]['sequence']

        unmask_sentence = unmask_sentence.replace("  ", " ")

        return unmask_sentence