import re
from bs4 import BeautifulSoup
from selenium import webdriver
import random
import requests
from kiwipiepy import Kiwi
import time
from quickspacer import Spacer
from typing import List

class AdverbAugmentation():
    def __init__(self):
        self.kiwi = Kiwi()
        self.spacer = Spacer()
        
    def _adverb_detector(self, sentence: str) -> List[str]:
        """
        Args:
            sentence (str): 문장
        Returns:
            adverbs (list): 부사 목록
        토큰화된 문장에서 부사를 검출합니다."""
        pos_list = self.kiwi.tokenize(sentence)
        adverbs = [token.form for token in pos_list if token.tag == "MAG" and len(token.form) > 1]
        return adverbs

    def _get_gloss(self, word: str) -> str:
        """
        온라인 사전에서 단어의 사전적 의미를 찾습니다.
        Args:
            word (str): 단어

        Returns:
            gloss_corrected (str): 띄어쓰기 교정까지 완료된 사전적 의미
        """
        url = f"https://dic.daum.net/search.do?q={word}"
        try:
            res = requests.get(url, timeout=5)
            soup = BeautifulSoup(res.content, "html.parser")
            # 첫번째 의미 추출
            meaning = soup.find('span', class_='txt_search')
            if not meaning:
                return word
            # 추출된 의미에서 한글만 추출
            gloss = ' '.join(re.findall('[가-힣]+', str(meaning)))
            # 띄어쓰기 교정
            gloss = gloss.replace(" ", "")
            
            # 검색된 의미가 없다면 원 단어 반환
            if not gloss.strip(): 
                return word
            
            gloss_corrected = self.spacer.space([gloss])
            return gloss_corrected[0].strip()
        except (requests.RequestException, AttributeError):
            print(f"Failed to fetch gloss for word: {word}")
            return word
    
    def adverb_gloss_replacement(self, sentence: str) -> str:
        """
        문장에서 무작위로 선택된 부사를 그것의 사전적 의미로 교체합니다.

        Args:
            sentence (str): 문장

        Returns:
            sentence (str): 교체된 문장
        """
        
        adverbs = self._adverb_detector(sentence)
        if adverbs:
            # Randomly select one adverb for replacement
            adverb_to_replace = random.choice(adverbs)
            gloss = self._get_gloss(adverb_to_replace)
            return sentence.replace(adverb_to_replace, gloss)
        return sentence