"""
func 1. adverb to gloss

"""
import re
from bs4 import BeautifulSoup
from selenium import webdriver
import random
import requests
from kiwipiepy import Kiwi
import time
from hanspell import spell_checker

class AdverbAugmentation():
    def __init__(self):
        self.kiwi = Kiwi()

    def _adverb_detector(self, sentence):

        # POS info
        pos_list = [(x[0], x[1]) for x in self.kiwi.tokenize(sentence)] # (token, pos)
        
        adverb_list = []
        for pos in pos_list:
            if pos[1] == "MAG" and len(pos[0]) > 1: # 1음절 부사는 제외함.
                adverb_list.append(pos[0])
        return adverb_list

    def _get_gloss(self, word):
        res = requests.get("https://dic.daum.net/search.do?q=" + word, timeout=5)
        time.sleep(random.uniform(2,4))
        soup = BeautifulSoup(res.content, "html.parser")
        try:
            # 첫 번째 뜻풀이.
            meaning = soup.find('span', class_='txt_search')
        except AttributeError:
            return word
        if meaning == None:
            return word
        
        # parsing 결과에서 한글만 추출
        meaning = re.findall('[가-힣]+', str(meaning))
        meaning = ' '.join(meaning)
        
        # 띄어쓰기 오류 교정 (위 에 -> 위에)
        meaning = spell_checker.check(meaning).as_dict()['checked'].strip()
        return meaning.strip()
    
    def adverb_gloss_replacement(self, sentence):
        print(sentence)
        adverb_list = self._adverb_detector(sentence)
        if adverb_list:
            # 부사들 중에서 1개만 랜덤으로 선택합니다.
            adverb = random.choice(adverb_list)
            gloss = self._get_gloss(adverb)
            sentence = sentence.replace(adverb, gloss)
        return sentence
        

# adverb_aug = AdverbAugmentation()

# sentence = "눈이 굉장히 천천히, 그리고 아주 조금씩 하얗게 쌓이고 있다."

# result = adverb_aug.adverb_gloss_replacement(sentence)
# print('input: ', sentence)
# print('reuslt: ', result)