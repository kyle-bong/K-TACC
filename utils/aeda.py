import random

def aeda(sentence: str) -> str:
    """AEDA: Randomly inserts punctuation into a sentence to augment data.

    Args:
        sentence (str): Input sentence.

    Returns:
        str: Augmented sentence with random punctuation inserted.
    """
    punc_list = list(".,;:?!")
    words = sentence.split()
    
    if len(words) < 2:
        return sentence

    random_ratio = random.uniform(0.1, 0.3) # AEDA 논문을 따름
    n_punc_to_insert = max(1, int(len(words) * random_ratio))

    for _ in range(n_punc_to_insert):
        random_punc = random.choice(punc_list)
        random_idx = random.randint(1, len(words) - 1)
        words.insert(random_idx, random_punc)

    return ' '.join(words).strip()
