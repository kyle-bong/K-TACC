# Making Augset
import pandas as pd
from BERT_augmentation import BERT_Augmentation
from adverb_augmentation import AdverbAugmentation
from aeda import aeda
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import numpy as np
tqdm.pandas()

### TODO: multi processing (kaggle 코드 참고)

BERT_aug = BERT_Augmentation()
random_masking_replacement = BERT_aug.random_masking_replacement
random_masking_insertion = BERT_aug.random_masking_insertion
adverb_aug = AdverbAugmentation()
adverb_gloss_replacement = adverb_aug.adverb_gloss_replacement

orig_train = pd.read_json('sts/datasets/klue-sts-v1.1_train.json')

# multi processing
def parallelize_dataframe(df, func, num_cores):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def apply_random_masking_replacement(df):
    df['sentence1'] = df['sentence1'].progress_apply(lambda x: random_masking_replacement(x, span_ratio=0.15))
    df['sentence2'] = df['sentence2'].progress_apply(lambda x: random_masking_replacement(x, span_ratio=0.15))
    return df


# dataset
random_masking_replacement_train = parallelize_dataframe(orig_train.copy(), apply_random_masking_replacement, 4)

# span_ratio = 0.15
# random_masking_replacement_train = orig_train.copy()
# random_masking_replacement_train['sentence1'] = random_masking_replacement_train['sentence1'].progress_apply(lambda x: random_masking_replacement(x, span_ratio=span_ratio))# if len(x.split()) > 1//span_ratio else x)
# random_masking_replacement_train['sentence2'] = random_masking_replacement_train['sentence2'].progress_apply(lambda x: random_masking_replacement(x, span_ratio=span_ratio))# if len(x.split()) > 1//span_ratio else x)
random_masking_replacement_augset = pd.concat([orig_train, random_masking_replacement_train])
random_masking_replacement_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
print(len(random_masking_replacement_augset))

random_masking_replacement_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_random_masking_replacement_augset_span_0.15.json')

# random masking (span_ratio=0.30)
# span_ratio=0.30
# random_masking_replacement_train = orig_train.copy()
# random_masking_replacement_train['sentence1'] = random_masking_replacement_train['sentence1'].progress_apply(lambda x: random_masking_replacement(x, span_ratio=2)) #if len(x.split()) > 1//span_ratio else random_masking_replacement(x, span_ratio=span_ratio/2) if len(x.split()) > 1//span_ratio else x)
# random_masking_replacement_train['sentence2'] = random_masking_replacement_train['sentence2'].progress_apply(lambda x: random_masking_replacement(x, span_ratio=2)) #if len(x.split()) > 1//span_ratio else random_masking_replacement(x, span_ratio=span_ratio/2) if len(x.split()) > 1//span_ratio else x)
# random_masking_replacement_augset = pd.concat([orig_train, random_masking_replacement_train])
# random_masking_replacement_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(random_masking_replacement_augset))

# random_masking_replacement_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_random_masking_replacement_augset_span_0.30.json')

# # random insertion (span_ratio=0.15)
# span_ratio = 0.15
# random_masking_insertion_train = orig_train.copy()
# random_masking_insertion_train['sentence1'] = random_masking_insertion_train['sentence1'].progress_apply(lambda x: random_masking_insertion(x, span_ratio=span_ratio))
# random_masking_insertion_train['sentence2'] = random_masking_insertion_train['sentence2'].progress_apply(lambda x: random_masking_insertion(x, span_ratio=span_ratio))
# random_masking_insertion_augset = pd.concat([orig_train, random_masking_insertion_train])
# random_masking_insertion_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(random_masking_insertion_augset))
# random_masking_insertion_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_random_masking_insertion_augset_span_0.15.json')

# # random insertion (span_ratio=0.30)
# span_ratio=0.30
# random_masking_insertion_train = orig_train.copy()
# random_masking_insertion_train['sentence1'] = random_masking_insertion_train['sentence1'].progress_apply(lambda x: random_masking_insertion(x, span_ratio=span_ratio))
# random_masking_insertion_train['sentence2'] = random_masking_insertion_train['sentence2'].progress_apply(lambda x: random_masking_insertion(x, span_ratio=span_ratio))
# random_masking_insertion_augset = pd.concat([orig_train, random_masking_insertion_train])
# random_masking_insertion_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(random_masking_insertion_augset))
# random_masking_insertion_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_random_masking_insertion_augset_span_0.30.json')

# # adverb
# adverb_train = orig_train.copy()
# adverb_train['sentence1'] = adverb_train['sentence1'].progress_apply(lambda x: adverb_gloss_replacement(x))
# adverb_train['sentence2'] = adverb_train['sentence2'].progress_apply(lambda x: adverb_gloss_replacement(x))
# adverb_augset = pd.concat([orig_train, adverb_train])
# adverb_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(adverb_augset))
# adverb_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_adverb_augset.json')


# koreda
from koreda import synonym_replacement, random_deletion, random_swap, random_insertion

# synonym_replacement
# sr_train = orig_train.copy()
# sr_train['sentence1'] = sr_train['sentence1'].apply(lambda x: " ".join(synonym_replacement(x.split(), 1)))
# sr_train['sentence2'] = sr_train['sentence2'].apply(lambda x: " ".join(synonym_replacement(x.split(), 1)))
# sr_augset = pd.concat([orig_train, sr_train])
# sr_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(sr_augset))
# sr_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_sr_augset.json')

# # # random_deletion
# rd_train = orig_train.copy()
# rd_train['sentence1'] = rd_train['sentence1'].apply(lambda x: " ".join(random_deletion(x.split(), 1)))
# rd_train['sentence2'] = rd_train['sentence2'].apply(lambda x: " ".join(random_deletion(x.split(), 1)))
# rd_augset = pd.concat([orig_train, rd_train])
# rd_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(rd_augset))
# rd_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_rd_augset.json')

# # # random_swap
# rs_train = orig_train.copy()
# rs_train['sentence1'] = rs_train['sentence1'].apply(lambda x: " ".join(random_swap(x.split(), 1)))
# rs_train['sentence2'] = rs_train['sentence2'].apply(lambda x: " ".join(random_swap(x.split(), 1)))
# rs_augset = pd.concat([orig_train, rs_train])
# rs_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(rs_augset))
# rs_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_rs_augset.json')

# # # random_insertion
# ri_train = orig_train.copy()
# ri_train['sentence1'] = ri_train['sentence1'].apply(lambda x: " ".join(random_insertion(x.split(), 1)))
# ri_train['sentence2'] = ri_train['sentence2'].apply(lambda x: " ".join(random_insertion(x.split(), 1)))
# ri_augset = pd.concat([orig_train, ri_train])
# ri_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(ri_augset))
# ri_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_ri_augset.json')

# # aeda
# aeda_train = orig_train.copy()
# aeda_train['sentence1'] = aeda_train['sentence1'].apply(lambda x: " ".join(random_insertion(x.split(), 1)))
# aeda_train['sentence2'] = aeda_train['sentence2'].apply(lambda x: " ".join(random_insertion(x.split(), 1)))
# aeda_augset = pd.concat([orig_train, aeda_train])
# aeda_augset.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
# print(len(aeda_augset))
# aeda_augset.reset_index().to_json('sts/datasets/klue-sts-v1.1_train_aeda_augset.json')



