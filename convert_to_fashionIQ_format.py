import argparse
import copy
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import nltk
import numpy as np
from nltk.corpus import stopwords
from PIL import Image, ImageFile

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def copy_move_img_and_rename(filepath):
    product_folders = os.listdir(filepath)
    for product in product_folders:
        for img in os.listdir(filepath / product / "images"):
            shutil.copyfile(filepath / product / "images" / img, filepath / product / (product + "_" + img))
    

def src_folder_to_dict_of_file_path_and_title(filepath):
    product_folders = os.listdir(filepath)
    product_file_path_and_title = []
    for product in product_folders:
        img_paths = []
        for img in os.listdir(filepath / product):
            if os.path.isfile(filepath / product / img):
                # get filename and extension
                filename, file_extension = os.path.splitext(img)
                if (file_extension == ".jpg" or file_extension == ".jpeg" or file_extension == ".png") and filename[-1] == "0":
                    img_paths.append(filepath / product / img)
        # read title.txt file
        title = ""
        with open(filepath / product / "title.txt", "r", encoding="utf8") as f:
            title = f.read()
            f.close()
        # read description.txt file
        description = ""
        with open(filepath / product / "description.txt", "r", encoding="utf8") as f:
            description = f.read()
            f.close()
        category = ""
        with open(filepath / product / "category.txt", "r", encoding="utf8") as f:
            category = f.read()
            f.close()
        # category = "general"
        for img_path in img_paths:
            product_file_path_and_title.append([img_path, title, description, category])
    return product_file_path_and_title


if __name__ == "__main__":

    # params
    # src_folder = Path("./adidas1600_train")
    # src_folder = Path("./adidas400_val")
    # src_folder = Path("./addidasWithCate")
    src_folder = Path("./adidas2000_val")

    dst_folder = Path("./fashionIQ")

    d = {'T-Shirts & Polos': 'tshirt_and_polo', 'Shorts': 'short', 'SportSwears': 'sportswear', 'Jackets': 'jacket', 'T-Shirts and Tops': 'tshirt_and_top', 'Dresses': 'dress', 'Skirts': 'skirt', 'Leggings': 'legging', 'Jerseys': 'jersey', 'Tracksuit': 'tracksuit', 'Hoodies': 'hoodie', 'Pants': 'pant', 'Tights': 'tight'}
    # d = {"general": "general"}


    copy_move_img_and_rename(Path(src_folder))
    img_path_list = src_folder_to_dict_of_file_path_and_title(src_folder)
    res = []
    for item in img_path_list:
        if item[1] in res:
            continue
        else:
            res.append(item[1])
    print(len(res))
    print(len(img_path_list))
    print("#"*80)
    # print(img_path_list)
    # img_path, title, description, category
    img_path_list_origin = copy.deepcopy(img_path_list)
    # partition img_path_list by category
    datasetTypes = ["train", "val"]
    for keys in d.keys():
        temp_list = []
        for item in img_path_list_origin:
            if item[3] == keys:
                temp_list.append(item)
        print(keys, len(temp_list))
        # print(temp_list)
        # random.shuffle(temp_list)
        for datasetType in datasetTypes:
            split_limit = int(0.7*len(temp_list))
            if datasetType == "train":
                img_path_list = temp_list[:split_limit]
            elif datasetType == "val":
                img_path_list = temp_list[split_limit:]
            # print(img_path_list)

            cap_json = []
            for i in range(len(img_path_list)):
                for j in range(i + 1, len(img_path_list)):
                    # get img name
                    img_name_i = os.path.basename(img_path_list[i][0]).split('/')[-1]
                    img_name_i = img_name_i.split('.')[0]
                    img_name_j = os.path.basename(img_path_list[j][0]).split('/')[-1]
                    img_name_j = img_name_j.split('.')[0]
                    # print(img_name_i)
                    if img_path_list[i][1] != img_path_list[j][1] \
                        and img_path_list[i][3] == keys and img_path_list[j][3] == keys: # same category
                        cap_json.append({"target": img_name_j, "candidate": img_name_i, "captions": ["change from " + img_path_list[i][1] + " to " + img_path_list[j][1], ""]})
            if datasetType == "train":
                limit = 1000
            elif datasetType == "val":
                limit = 200
            if len(cap_json) > limit:
                cap_json = cap_json[:limit]
            print(len(cap_json))

            # dst_folder / "captions"
            if not os.path.exists(dst_folder / "captions"):
                os.makedirs(dst_folder / "captions")
            captions_filename = "cap." + d[keys] + "." + datasetType + ".json"
            if not os.path.exists(dst_folder / "captions" / captions_filename):
                open(dst_folder / "captions" / captions_filename, "x").close()
            with open(dst_folder / "captions" / captions_filename, "w") as f:
                f.write(json.dumps(cap_json))

            # dst_folder / "image_splits"
            split_json = []
            for i in range(len(img_path_list)):
                if img_path_list[i][3] == keys:
                    # get img name
                    img_name_i = os.path.basename(img_path_list[i][0]).split('/')[-1]
                    img_name_i = img_name_i.split('.')[0]
                    split_json.append(img_name_i)
            if not os.path.exists(dst_folder / "image_splits"):
                os.makedirs(dst_folder / "image_splits")
            image_splits_filename = "split." + d[keys] + "." + datasetType + ".json"
            if not os.path.exists(dst_folder / "image_splits" / image_splits_filename):
                open(dst_folder / "image_splits" / image_splits_filename, "x").close()
            with open(dst_folder / "image_splits" / image_splits_filename, "w") as f:
                f.write(json.dumps(split_json))

            # # dst_folder / "images"
            # if not os.path.exists(dst_folder / "images"):
            #     os.makedirs(dst_folder / "images")
            # for img_title in img_path_list:
            #     img_name = os.path.basename(img_title[0]).split('/')[-1]
            #     # img_name = img_name.split('.')[0]
            #     shutil.copyfile(img_title[0], dst_folder / "images" / img_name)

            # dst_folder / "tags"
            asin2attr_json = []
            stop_words = set(stopwords.words('english'))
            for i in range(len(img_path_list)):
                if img_path_list[i][3] == keys:
                    img_name_i = os.path.basename(img_path_list[i][0]).split('/')[-1]
                    img_name_i = img_name_i.split('.')[0]
                    # tokenize description
                    description = img_path_list[i][2].lower()
                    # get substring before "product code:"
                    description = description[:description.find("product code:")]
                    description_tokens = nltk.word_tokenize(description)
                    # POS tagging
                    tagged_description_tokens = nltk.pos_tag(description_tokens)
                    # print(tagged_description_tokens)
                    description_tokens = []
                    # filter to take only nouns and adjectives
                    for tag in tagged_description_tokens:
                        if tag[1] == 'NN' or tag[1] == 'NNS' or tag[1] == 'NNP' or tag[1] == 'NNPS' or tag[1] == 'JJ':
                            description_tokens.append(tag[0])
                    # print(description_tokens)

                    # filter out stop words
                    description_tokens = [word for word in description_tokens if word not in stop_words]
                    # filter out punctuation
                    description_tokens = [word for word in description_tokens if word.isalnum()]
                    # filter out duplicate words
                    description_tokens = list(set(description_tokens))
                    
                    asin2attr_json.append({img_name_i : description_tokens})
            if not os.path.exists(dst_folder / "tags"):
                os.makedirs(dst_folder / "tags")
            tag_filename = "asin2attr." + d[keys] + "." + datasetType + ".json"
            if not os.path.exists(dst_folder / "tags" / tag_filename):
                open(dst_folder / "tags" / tag_filename, "x").close()
            with open(dst_folder / "tags" / tag_filename, "w") as f:
                f.write(json.dumps(asin2attr_json))
