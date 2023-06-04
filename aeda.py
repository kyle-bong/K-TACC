import random

def aeda(sentence):
    punc_list = list(".,;:?!")
    
    sentence = sentence.split()
    random_ratio = random.uniform(0.1, 0.3) # 범위는 ADEA 논문을 따름.
    n_ri = max(1, int(len(sentence) * random_ratio))
    
    for _ in range(n_ri):
        random_punc = random.choice(punc_list)
        random_idx = random.randint(0, len(sentence)-1)
        sentence.insert(random_idx, random_punc)
        
    return ' '.join(sentence).strip()
