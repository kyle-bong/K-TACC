import re
from bs4 import BeautifulSoup
from selenium import webdriver
import random
import requests
from kiwipiepy import Kiwi
import time
from quickspacer import Spacer
class AdverbAugmentation():
    def __init__(self):
        self.kiwi = Kiwi()
        self.spacing = Spacer().space
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
        time.sleep(random.uniform(0.5,2.5))
        soup = BeautifulSoup(res.content, "html.parser")
        try:
            # 첫 번째 뜻풀이.
            meaning = soup.find('span', class_='txt_search')
        except AttributeError:
            return word
        if not meaning:
            return word
        
        # parsing 결과에서 한글만 추출
        meaning = re.findall('[가-힣]+', str(meaning))
        meaning = ' '.join(meaning)
        
        # 띄어쓰기 오류 교정 (위 에 -> 위에)
        # meaning = spell_checker.check(meaning).as_dict()['checked'].strip()
        meaning = self.spacing([meaning.replace(" ", "")])
        return meaning[0].strip()
    
    def adverb_gloss_replacement(self, sentence):
        adverb_list = self._adverb_detector(sentence)
        if adverb_list:
            # 부사들 중에서 1개만 랜덤으로 선택합니다.
            adverb = random.choice(adverb_list)
            try:
                gloss = self._get_gloss(adverb)
                sentence = sentence.replace(adverb, gloss)
            except:
                print('except: ', sentence)
                pass
        return sentence