import pandas as pd
from utils.BertAugmentation import BertAugmentation
from utils.adverb_augmentation import AdverbAugmentation
from utils.LlmAugmentation import LlmAugmentation
from utils.aeda import aeda
from utils.koreda import synonym_replacement, random_deletion, random_swap, random_insertion
from tqdm import tqdm
import joblib
from functools import partial

tqdm.pandas()

BERT_aug = BertAugmentation()
adverb_aug = AdverbAugmentation()

def apply_augmentation(df, aug_func, n_jobs=8):
    """augmentation 병렬 처리"""
    pool = joblib.Parallel(n_jobs=n_jobs, prefer='threads')
    mapper = joblib.delayed(aug_func)
    
    tasks_1 = [mapper(row) for row in df['sentence1']]
    tasks_2 = [mapper(row) for row in df['sentence2']]
    
    df['sentence1'] = pool(tqdm(tasks_1))
    df['sentence2'] = pool(tqdm(tasks_2))
    
    return df

def apply_llm_augmentation(df, batch_size=5):
    """LlmAugmentation을 배치 단위로 적용합니다.

    Args:
        df (pd.DataFrame): 데이터셋
        batch_size (int): 배치 당 문장 개수

    Returns:
        df_aug (pd.DataFrame): 증강된 데이터셋
    """
    llm_aug = LlmAugmentation()
    df_aug = df.copy()

    sentences1 = df['sentence1'].tolist()
    augmented_sentences1 = []
    for i in tqdm(range(0, len(sentences1), batch_size)):
        batch = sentences1[i:i + batch_size]
        try:
            paraphrased_batch = llm_aug.generate_paraphrased_sentence(batch)
            augmented_sentences1.extend(paraphrased_batch)
        except Exception as e:
            print(f"Error processing batch {i} of sentence1: {e}")
            augmented_sentences1.extend(batch) 
    sentences2 = df['sentence2'].tolist()
    augmented_sentences2 = []
    for i in tqdm(range(0, len(sentences2), batch_size)):
        batch = sentences2[i:i + batch_size]
        try:
            paraphrased_batch = llm_aug.generate_paraphrased_sentence(batch)
            augmented_sentences2.extend(paraphrased_batch)
        except Exception as e:
            print(f"Error processing batch {i} of sentence2: {e}")
            augmented_sentences2.extend(batch)

    df_aug['sentence1'] = augmented_sentences1
    df_aug['sentence2'] = augmented_sentences2

    return df_aug

def save_augmented_dataset(df, filename):
    """중복 제거 후 증강된 데이터셋 저장"""
    df.drop_duplicates(['sentence1', 'sentence2'], inplace=True)
    df.reset_index(drop=True).to_json(filename)

# KoreDA augmentations
def apply_koreda_augmentation(df, aug_func, name, ratio=0.15):
    """Applies a KoreDA augmentation and saves the dataset."""
    df_copy = df.copy()
    df_copy['sentence1'] = df_copy['sentence1'].apply(lambda x: " ".join(aug_func(x.split(), ratio)))
    df_copy['sentence2'] = df_copy['sentence2'].apply(lambda x: " ".join(aug_func(x.split(), ratio)))
    save_augmented_dataset(pd.concat([orig_train, df_copy]), f'sts/datasets/klue-sts-v1.1_train_{name}_augset.json')

# Random masking replacement
def random_masking_replacement(sentence, ratio=0.15):
    return BERT_aug.random_masking_replacement(sentence, ratio=ratio)

# Random masking insertion
def random_masking_insertion(sentence, ratio=0.15):
    return BERT_aug.random_masking_insertion(sentence, ratio=ratio)


if __name__ == "__main__":
    orig_train = pd.read_json('sts/datasets/klue-sts-v1.1_train.json')

    # Apply random masking replacement
    random_masking_replacement_train = orig_train.copy()
    random_masking_replacement_train = apply_augmentation(random_masking_replacement_train, partial(random_masking_replacement, ratio=0.15))
    save_augmented_dataset(pd.concat([orig_train, random_masking_replacement_train]), 'sts/datasets/klue-sts-v1.1_train_random_masking_replacement_augset.json')

    # Apply random masking insertion
    random_masking_insertion_train = orig_train.copy()
    random_masking_insertion_train = apply_augmentation(random_masking_insertion_train, partial(random_masking_insertion, ratio=0.15))
    save_augmented_dataset(pd.concat([orig_train, random_masking_insertion_train]), 'sts/datasets/klue-sts-v1.1_train_random_masking_insertion_augset.json')

    # Apply adverb gloss replacement
    adverb_train = orig_train.copy()
    adverb_train['sentence1'] = adverb_train['sentence1'].progress_apply(lambda x: adverb_aug.adverb_gloss_replacement(x))
    adverb_train['sentence2'] = adverb_train['sentence2'].progress_apply(lambda x: adverb_aug.adverb_gloss_replacement(x))
    save_augmented_dataset(pd.concat([orig_train, adverb_train]), 'sts/datasets/klue-sts-v1.1_train_adverb_augset.json')

    apply_koreda_augmentation(orig_train, synonym_replacement, 'sr', ratio=1)
    apply_koreda_augmentation(orig_train, random_deletion, 'rd', ratio=0.15)
    apply_koreda_augmentation(orig_train, random_swap, 'rs', ratio=1)
    apply_koreda_augmentation(orig_train, random_insertion, 'ri', ratio=1)

    # Apply AEDA augmentation
    aeda_train = orig_train.copy()
    aeda_train['sentence1'] = aeda_train['sentence1'].progress_apply(aeda)
    aeda_train['sentence2'] = aeda_train['sentence2'].progress_apply(aeda)
    save_augmented_dataset(pd.concat([orig_train, aeda_train]), 'sts/datasets/klue-sts-v1.1_train_aeda_augset.json')

    # Apply LLM augmentation
    # 실행 전 과금 여부에 주의하시기 바랍니다.
    llm_train = orig_train.copy()
    
    # batch inference를 통해 전체 요청 수를 줄입니다.
    llm_augmented_train = apply_llm_augmentation(orig_train, batch_size=32)
    save_augmented_dataset(pd.concat([orig_train, llm_augmented_train]), 'sts/dataset/klue-sts-v1.1_train_llm_augset.json')