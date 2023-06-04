# 문맥을 고려한 한국어 텍스트 데이터 증강 (Korean Text Augmentation Considering Context, **K-TACC**)

## Overview
- `BERT_augmentation` : BERT based 모델을 활용하여, 의미상 자연스러운 토큰을 삽입하거나 대체하는 형식으로 문장 augmentation 수행
- `Adverb_augmentation`: 부사를 그 부사의 뜻풀이로 교체하는 형식으로 문장 augmentation 수행
- 기존 EDA(Easy Data Augmentation), AEDA(An Easier Data Augmentation) 기법에 비해 의미적으로 좀 더 자연스러운 문장 생성 가능

## Usage
### 설치
```
bash install.sh
```

### Pilot test
`augmentation.ipynb` 파일에서 여러 증강 기법의 샘플을 확인해보실 수 있습니다.

### 증강 데이터 생성
실행 전 데이터셋의 경로, 저장할 파일 이름을 직접 지정해야 합니다.
Adverb augmentation의 경우 사전 뜻풀이를 크롤링해오기 때문에 시간이 오래걸릴 수 있습니다.
증강 적용 후 원본 데이터셋과 중복이 발생할 수 있으므로 중복을 제거하는 과정을 거칩니다. 이 때문에 각 증강 방법마다 결과로 나오는 데이터셋의 크기가 서로 다를 수 있습니다. 
```
python augmentation.py
```

### STS 성능 평가
실행 전 wandb login이 필요합니다.
```
cd sts
bash train.sh
```

## Experiment
문맥을 고려하여 [MASK] 토큰을 복원할 수 있는 BERT 기반 모델이라면 좀 더 자연스러운 증강이 가능할 것이라는 가설 하에 실험을 진행하였습니다. 각 Augmentation 기법의 성능 실험을 위해, 문장의 의미를 민감하게 파악하여야 하는 Semantic Text Similarity (STS)를 진행하였습니다. base model은 RoBERTa-base(`klue/roberta-base`) 모델로 선정하였고, 데이터셋은 KLUE STS 데이터셋을 사용하였습니다.
실험한 증강 기법은 본 repository에서 제안하는 BERT Augmentation 중에서 특정 단어를 maksing한 뒤 다시 복원하는 Random Masking Replacement 기법 및 문장에 [mask] 토큰을 추가하고 이를 복원하는 Random Masking Insertion을 실험하였으며, 기존에 제안되었던 EDA (Easy Data Augmentation), AEDA (An Easier Data Augmentation)도 함께 실험하였습니다.

|Model|Pearson's correlation|
|---|---|
|base|0.9232|
|EDA (Random Deletion) | 0.8960|
|EDA (Random Swap) | 0.9243 |
|EDA (Random Synonym Replacement) | 0.9250 |
|EDA (Random Insertion | 0.9259 |
|AEDA | 0.9252 |
|Adverb augmentation | 0.9299 |
|BERT_Augmentation (Random Masking Replacement) | 0.9023 |
|BERT_Augmentation (Random Masking Insertion) | **0.9300** |

실험 결과, BERT_Augmentation (Random Masking Insertion)이 가장 성능이 높게 나타났습니다. Adverb augmentation, EDA (Random Insertion), AEDA, EDA (Random Synonym Replacement), EDA (Random Swap)도 base에 비해 높은 성능을 보였습니다. 한편, EDA (Random Deletion), BERT_Augmentation (Random Masking Replacement) 방식은 base 모델보다 성능이 떨어지는 것으로 나타났습니다. 이 두 방식은 문장 내에서 단어를 무작위로 선택하여 삭제하거나 다른 단어로 교체한다는 점에서, 문장 내에서 핵심적인 의미를 지니는 단어를 훼손할 가능성이 있습니다. 반면 성능이 좋게 나온 BERT Augmentation (Random Masking Insertion), AEDA, EDA (Random Insertion) 방식은 원본 문장의 단어는 그대로 보존한 채 단어나 기호를 추가하는 방식이기 때문에 성능이 좋게 나온 것으로 보입니다.
그리고 Adverb augmentation의 경우 문장에서 optional한 역할을 하는 부사를 바꿔주는 것이기 때문에 원본 문장의 의미 훼손이 적었을 것으로 판단됩니다.


## Reference
```
@article{karimi2021aeda,
  title={Aeda: An easier data augmentation technique for text classification},
  author={Karimi, Akbar and Rossi, Leonardo and Prati, Andrea},
  journal={arXiv preprint arXiv:2108.13230},
  year={2021}
}

@article{wei2019eda,
  title={Eda: Easy data augmentation techniques for boosting performance on text classification tasks},
  author={Wei, Jason and Zou, Kai},
  journal={arXiv preprint arXiv:1901.11196},
  year={2019}
}

https://github.com/catSirup/KorEDA
https://github.com/KLUE-benchmark/KLUE-baseline
```
