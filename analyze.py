import os
from pathlib import Path
import json
# read all files in folder
content = []
filepath = Path("./fashionIQ/captions/")
file_list = os.listdir(filepath)
for file in file_list:
    with open(filepath / file, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
        for item in data:
            sentence = item["captions"][0]
            for word in sentence.split(" "):
                content.append(word)  
        f.close()
# counter for each tag
tag_counter = {}
for tag in content:
    if tag in tag_counter:
        tag_counter[tag] += 1
    else:
        tag_counter[tag] = 1
# sort by count
tag_counter = dict(sorted(tag_counter.items(), key=lambda item: item[1], reverse=True))
tag_counter.pop("change")
tag_counter.pop("from")
tag_counter.pop("to")
print(tag_counter)
# export wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                min_font_size = 10).generate_from_frequencies(tag_counter)
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()
