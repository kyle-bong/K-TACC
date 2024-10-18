import google.generativeai as genai
import os
import json
from typing import List
import re

class LlmAugmentation():
    def __init__(
        self,
        temperature: float=1.0,
        candidate_count: int=1
    ):
        self.api_key = os.environ["GENAI_API_KEY"]
        genai.configure(api_key=self.api_key)
        
        self.system_instruction="당신은 뛰어난 언어 능력을 가진 문장가입니다."
        self.temperature=temperature
        self.candidate_count=candidate_count
        
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                candidate_count=self.candidate_count
            ),
            system_instruction=self.system_instruction
        )
        
        self.safety_settings={
            'HATE': 'BLOCK_NONE',
            'HARASSMENT': 'BLOCK_NONE',
            'SEXUAL' : 'BLOCK_NONE',
            'DANGEROUS' : 'BLOCK_NONE'
        }
        
    def generate_paraphrased_sentence(self, sentences: List[str]) -> List[str]:
        """문장 배열을 받아, 각 문장별로 의미는 동일하되 표현은 다른 문장으로 변행합니다. 

        Args:
            sentences (List[str]): 변형할 문장

        Returns:
            List[str]: 변형된 문장
        """
        
        prompt = """
            다음의 주어진 문장들을 의미는 동일하되 표현만 다르게 변형해주세요.
            결과는 JSON 형식의 문자열 리스트로만 출력해주세요.
            주의: 숫자와 고유명사(지명, 기관명, 인명 등)는 절대 바꾸지 마세요.
            
            아래의 예시는 참고용입니다.
            
            입력:
            [
                "오늘은 비가 많이 내린다.",
                "즐거운 추석 명절 보내시기 바랍니다.",
                "나랑 사귀자!",
                "피곤할 땐 아메리카노를 마시면 좋지.",
                "국토교통부가 디딤돌 대출 규제를 잠정 유예한다."
            ]

            출력:
            [
                "금일은 강수량이 많다."
                "한가위 명절 재미있게 보내세요."
                "우리 오늘부터 1일이야~"
                "졸리면 커피 한잔해~",
                "국토교통부가 디딤돌 대출 규제를 잠시 미루기로 결정하였다."  
            ]

            이 예시를 바탕으로 다음의 주어진 문장들의 표현을 다르게 변형해주세요.

            """ \
        + "\n".join(sentences)
        
        response = self.model.generate_content(
            prompt,
            safety_settings=self.safety_settings
        )
        
        try:
            response_text = response.text.strip()
            json_text = re.search(r'\[.*\]', response_text, re.DOTALL).group()
            paraphrased_sentences = json.loads(json_text)
            return paraphrased_sentences
        except Exception as e:
            print("Error parsing response:", e)
            print("Response text:", response_text)
            return sentences 
            
        
        
        
                
        