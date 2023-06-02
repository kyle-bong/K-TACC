# 문맥을 고려한 한국어 데이터 증강(Korean Text Augmentation Considering Context, KTACC)

## Overview
- 한국어 텍스트 데이터에 노이즈를 가해 데이터를 증강합니다. 이때 BERT based 모델을 활용하여, 원래의 문장과 문맥적으로 유사한 토큰을 삽입하거나 대체하는 형식으로 문장 augmentation을 수행하는 `BERT_augmentation`, 그리고 부사를 부사의 뜻풀이로 교체하여 문장을 변형하는 `Adverb_augmentation` 기법이 있습니다.
- 따라서, 기존 EDA(Easy Data Augmentation) 기법에 비해 의미적으로 좀 더 자연스러운 문장 생성이 가능합니다.

## Usage
### 증강 데이터 생성
```
python augmentation.py
```

### STS 성능 평가
```
cd sts
python train.sh
```

## Experiment



## Liscence

