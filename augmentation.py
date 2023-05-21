# Making Augset
import pandas as pd
from BERT_augmentation import BERT_Augmentation
from adverb_augmentation import AdverbAugmentation

BERT_aug = BERT_Augmentation()
random_masking_replacement = BERT_aug.random_masking_replacement
random_masking_insertion = BERT_aug.random_masking_insertion
adverb_aug = AdverbAugmentation()
adverb_gloss_replacement = adverb_aug.adverb_gloss_replacement

# dataset
orig_train = pd.read_json('sts/datasets/klue-sts-v1.1_train.json')
sen1_random_masking_replacement_train = orig_train.copy()
sen2_random_masking_replacement_train = orig_train.copy()
sen1_random_masking_replacement_train['sentence1'] = sen1_random_masking_replacement_train['sentence1'].apply(lambda x: random_masking_replacement(x, span=1) if len(x.split()) > 1 else x)
sen2_random_masking_replacement_train['sentence2'] = sen2_random_masking_replacement_train['sentence2'].apply(lambda x: random_masking_replacement(x, span=1) if len(x.split()) > 1 else x)
random_masking_replacement_augset = pd.concat([orig_train, sen1_random_masking_replacement_train, sen2_random_masking_replacement_train])
random_masking_replacement_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
print(len(random_masking_replacement_augset))

random_masking_replacement_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_random_masking_replacement_augset.json')


sen1_random_masking_insertion_train = orig_train.copy()
sen2_random_masking_insertion_train = orig_train.copy()
sen1_random_masking_insertion_train['sentence1'] = sen1_random_masking_insertion_train['sentence1'].apply(lambda x: random_masking_insertion(x))
sen2_random_masking_insertion_train['sentence2'] = sen2_random_masking_insertion_train['sentence2'].apply(lambda x: random_masking_insertion(x))
random_masking_insertion_augset = pd.concat([orig_train, sen1_random_masking_insertion_train, sen2_random_masking_insertion_train])
random_masking_insertion_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
print(len(random_masking_insertion_augset))
random_masking_insertion_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_random_masking_insertion_augset.json')

sen1_adverb_train = orig_train.copy()
sen2_adverb_train = orig_train.copy()
sen1_adverb_train['sentence1'] = sen1_adverb_train['sentence1'].apply(lambda x: adverb_gloss_replacement(x))
sen2_adverb_train['sentence2'] = sen2_adverb_train['sentence2'].apply(lambda x: adverb_gloss_replacement(x))
adverb_augset = pd.concat([orig_train, sen1_adverb_train, sen2_adverb_train])
adverb_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
print(len(adverb_augset))
adverb_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_adverb_augset.json')
