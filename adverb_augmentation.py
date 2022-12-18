    """
    func 1. adverb to gloss

    """

import re
from bs4 import BeautifulSoup
from selenium import webdriver
import random
import requests
import string
from kiwipiepy import Kiwi
import time
from py_hanspell.hanspell import spell_checker

class AdverbAugmentation():
    def __init__():
        self.kiwi = Kiwi()

    def _adverb_detector(self, sentence):
        # Tokenizing    
        tokenized = mecab.pos(sentence)

        # POS info
        pos_list = [(x[0], x[1]) for x in Kiwi.tokenize(sentence)] # (token, pos)
            
        for pos in pos_list:
            if pos[1] == "MAG" and len(pos[0]) > 1: # 1음절 부사는 제외함.
                return pos[0]

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
        # 문장 안에 부사가 존재한다면:
        adverb = self._adverb_detector(sentence)
        if adverb:
            gloss = self._get_gloss(adverb)
            return sentence.replace(adverb, gloss)