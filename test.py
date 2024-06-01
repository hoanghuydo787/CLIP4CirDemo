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
    print(stop_words)
    caption = caption.lower()
    for word in nltk.word_tokenize(caption):
        if word not in stop_words and \
        word.isalnum() and \
        process.extractOne(word, choices)[1] > 80:
            res.append(process.extractOne(word, choices))
    if len(res) == 0:
        return 'general'
    # sort the results by the score
    res.sort(key=lambda x: x[1], reverse=True)
    d = {'T-Shirts & Polos': 'tshirt_and_polo', 'Shorts': 'short', 'SportSwears': 'sportswear', 'Jackets': 'jacket', 'T-Shirts and Tops': 'tshirt_and_top', 'Dresses': 'dress', 'Skirts': 'skirt', 'Leggings': 'legging', 'Jerseys': 'jersey', 'Tracksuit': 'tracksuit', 'Hoodies': 'hoodie', 'Pants': 'pant', 'Tights': 'tight'}
    return d[res[0][0]]

# sentence = "i want it to be red and sleeveless with adidas logo on it"
# sentence = "I want a pink top for women"
# sentence = 'I WANT A PINK TOP FOR WOMEN'
sentence = "purpl dress with long sleeve"
# sentence = "purpl dress with long sleeve"
# sentence = "I want a jacket with darker color"
print(extract_category_from_caption(sentence))