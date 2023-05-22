# Making Augset
import pandas as pd
from BERT_augmentation import BERT_Augmentation
from adverb_augmentation import AdverbAugmentation
from aeda import aeda
from tqdm import tqdm
tqdm.pandas()

### TODO: multi processing (kaggle 코드 참고)

BERT_aug = BERT_Augmentation()
random_masking_replacement = BERT_aug.random_masking_replacement
random_masking_insertion = BERT_aug.random_masking_insertion
adverb_aug = AdverbAugmentation()
adverb_gloss_replacement = adverb_aug.adverb_gloss_replacement

orig_train = pd.read_json('sts/datasets/klue-sts-v1.1_train.json')

# dataset

# sen1_random_masking_replacement_train = orig_train.copy()
# sen2_random_masking_replacement_train = orig_train.copy()
# sen1_random_masking_replacement_train['sentence1'] = sen1_random_masking_replacement_train['sentence1'].progress_apply(lambda x: random_masking_replacement(x, span=1) if len(x.split()) > 1 else x)
# sen2_random_masking_replacement_train['sentence2'] = sen2_random_masking_replacement_train['sentence2'].progress_apply(lambda x: random_masking_replacement(x, span=1) if len(x.split()) > 1 else x)
# random_masking_replacement_augset = pd.concat([orig_train, sen1_random_masking_replacement_train, sen2_random_masking_replacement_train])
# random_masking_replacement_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(random_masking_replacement_augset))

# random_masking_replacement_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_random_masking_replacement_augset.json')


# sen1_random_masking_insertion_train = orig_train.copy()
# sen2_random_masking_insertion_train = orig_train.copy()
# sen1_random_masking_insertion_train['sentence1'] = sen1_random_masking_insertion_train['sentence1'].progress_apply(lambda x: random_masking_insertion(x))
# sen2_random_masking_insertion_train['sentence2'] = sen2_random_masking_insertion_train['sentence2'].progress_apply(lambda x: random_masking_insertion(x))
# random_masking_insertion_augset = pd.concat([orig_train, sen1_random_masking_insertion_train, sen2_random_masking_insertion_train])
# random_masking_insertion_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(random_masking_insertion_augset))
# random_masking_insertion_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_random_masking_insertion_augset.json')

# sen1_adverb_train = orig_train.copy()
# sen2_adverb_train = orig_train.copy()
# sen1_adverb_train['sentence1'] = sen1_adverb_train['sentence1'].progress_apply(lambda x: adverb_gloss_replacement(x))
# sen2_adverb_train['sentence2'] = sen2_adverb_train['sentence2'].progress_apply(lambda x: adverb_gloss_replacement(x))
# adverb_augset = pd.concat([orig_train, sen1_adverb_train, sen2_adverb_train])
# adverb_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(adverb_augset))
# adverb_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_adverb_augset.json')

# koreda
from koreda import synonym_replacement, random_deletion, random_swap, random_insertion

# synonym_replacement
# sen1_sr_train = orig_train.copy()
# sen2_sr_train = orig_train.copy()
# sen1_sr_train['sentence1'] = sen1_sr_train['sentence1'].apply(lambda x: " ".join(synonym_replacement(x.split(), 1)))
# sen2_sr_train['sentence2'] = sen2_sr_train['sentence2'].apply(lambda x: " ".join(synonym_replacement(x.split(), 1)))
# sr_augset = pd.concat([orig_train, sen1_sr_train, sen2_sr_train])
# sr_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(sr_augset))
# sr_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_sr_augset.json')

# # random_deletion
# sen1_rd_train = orig_train.copy()
# sen2_rd_train = orig_train.copy()
# sen1_rd_train['sentence1'] = sen1_rd_train['sentence1'].apply(lambda x: " ".join(random_deletion(x.split(), 1)))
# sen2_rd_train['sentence2'] = sen2_rd_train['sentence2'].apply(lambda x: " ".join(random_deletion(x.split(), 1)))
# rd_augset = pd.concat([orig_train, sen1_rd_train, sen2_rd_train])
# rd_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(rd_augset))
# rd_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_rd_augset.json')

# # random_swap
# sen1_rs_train = orig_train.copy()
# sen2_rs_train = orig_train.copy()
# sen1_rs_train['sentence1'] = sen1_rs_train['sentence1'].apply(lambda x: " ".join(random_swap(x.split(), 1)))
# sen2_rs_train['sentence2'] = sen2_rs_train['sentence2'].apply(lambda x: " ".join(random_swap(x.split(), 1)))
# rs_augset = pd.concat([orig_train, sen1_rs_train, sen2_rs_train])
# rs_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(rs_augset))
# rs_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_rs_augset.json')

# # random_insertion
# sen1_ri_train = orig_train.copy()
# sen2_ri_train = orig_train.copy()
# sen1_ri_train['sentence1'] = sen1_ri_train['sentence1'].apply(lambda x: " ".join(random_insertion(x.split(), 1)))
# sen2_ri_train['sentence2'] = sen2_ri_train['sentence2'].apply(lambda x: " ".join(random_insertion(x.split(), 1)))
# ri_augset = pd.concat([orig_train, sen1_ri_train, sen2_ri_train])
# ri_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(ri_augset))
# ri_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_ri_augset.json')

# aeda
sen1_aeda_train = orig_train.copy()
sen2_aeda_train = orig_train.copy()
sen1_aeda_train['sentence1'] = sen1_aeda_train['sentence1'].apply(lambda x: " ".join(random_insertion(x.split(), 1)))
sen2_aeda_train['sentence2'] = sen2_aeda_train['sentence2'].apply(lambda x: " ".join(random_insertion(x.split(), 1)))
aeda_augset = pd.concat([orig_train, sen1_aeda_train, sen2_aeda_train])
aeda_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
print(len(aeda_augset))
aeda_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_aeda_augset.json')



