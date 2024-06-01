import os
import json
from pathlib import Path

# read all json file in the directory with *.val.json
def read_val():
    filepath=Path('./fashionIQ/captions')
    # get all json file in the directory
    files = [f for f in os.listdir(filepath) if f.endswith('.val.json')]
    data = []
    for file in files:
        with open(filepath / file, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
            print(data)
    # write to a new json file
    with open('./fashionIQ/captions/cap.general.val.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    filepath=Path('./fashionIQ/image_splits')
    # get all json file in the directory
    files = [f for f in os.listdir(filepath) if f.endswith('.val.json')]
    data = []
    for file in files:
        with open(filepath / file, 'r', encoding='utf-8') as f:
            data.extend(json.load(f))
            print(data)
    # write to a new json file
    with open('./fashionIQ/image_splits/split.general.val.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


read_val()