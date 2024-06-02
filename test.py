from thefuzz import fuzz
from thefuzz import process
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_category_from_caption(caption: str) -> str:
    choices = ['T-Shirts & Polos', 'Shorts', 'SportSwears', 'Jackets', 'T-Shirts and Tops', 'Dresses', 'Skirts', 'Leggings', 'Jerseys', 'Tracksuit', 'Hoodies', 'Pants', 'Tights']
    res = []
    stop_words = set(stopwords.words('english'))
    caption = caption.lower()
    for word in nltk.word_tokenize(caption):
        if word not in stop_words and \
        process.extractOne(word, choices)[1] > 75:
            res.append(process.extractOne(word, choices))
    if len(res) == 0:
        return 'general'
    # sort the results by the score
    res.sort(key=lambda x: x[1], reverse=True)
    d = {'T-Shirts & Polos': 'tshirt_and_polo', 'Shorts': 'short', 'SportSwears': 'sportswear', 'Jackets': 'jacket', 'T-Shirts and Tops': 'tshirt_and_top', 'Dresses': 'dress', 'Skirts': 'skirt', 'Leggings': 'legging', 'Jerseys': 'jersey', 'Tracksuit': 'tracksuit', 'Hoodies': 'hoodie', 'Pants': 'pant', 'Tights': 'tight'}
    return d[res[0][0]]

test_cases = [
    ("I need an outfit for a formal event.", "general"),
    ("Could you find me a pair of running leggings?", "legging"),
    ("How about a top or maybe a jacket?", ["tshirt_and_top", "jacket"]),
    ("Can you get me a hoodie, please?", "hoodie"),
    ("Find me a unicorn costume.", "general"),
    ("I'm looking for Nike tracksuit.", "tracksuit"),
    ("I need pants and a hoodie.", ["pant", "hoodie"]),
    ("i want it to be red and sleeveless with adidas logo on it", "general"),
    ("I want a pink top for women", "tshirt_and_top"),
    ("I WANT A PINK TOP FOR WOMEN", "tshirt_and_top"),
    ("purpl dress with long sleeve", "dress"),
    ("I want a jacket with darker color", "jacket"),
    ("I'm looking for some slacks to go with my shirt.", ["tshirt_and_polo", "tshirt_and_top"]),
    ("I'd like a dreess for the party.", "dress"),    
    ("T-SHIRT! I need a T-shirt for my brother.", "tshirt_and_polo"),
    ("I need a new jacket, some tights, and maybe a jersey.", ["jacket", "tight", "jersey"]),
    ("Looking for something to keep me warm in winter, perhaps a hoodie.", "hoodie"),
    ("Quiero un vestido para la fiesta.", "general"),
    ("I'm searching for a pair of PJs.", "general"),
    ("A skirt that fits well would be nice.", "skirt"),
    ("I'm looking for a sleeveless dress with floral patterns.", "dress"),
    ("Not a fan of hoodies, maybe a tracksuit.", ["tracksuit", "hoodie"]),
    ("Need some comfy sweats for lounging.", "general"),
    ("I'd rather have leggings than shorts.", ["legging", "short"]),
    ("T-Shirts and shorts for summer, please.", ["tshirt_and_polo", "short"]),
]

for i, (sentence, expected) in enumerate(test_cases):
    result = extract_category_from_caption(sentence)
    if isinstance(expected, list):
        assert result in expected, f"Test case {i+1} failed: Expected one of {expected}, but got {result}"
    else:
        assert result == expected, f"Test case {i+1} failed: Expected {expected}, but got {result}"
    print(f"Test case {i+1} passed")

print("All test cases passed.")
