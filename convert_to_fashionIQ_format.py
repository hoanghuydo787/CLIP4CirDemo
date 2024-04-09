import argparse
import json
import logging
import os
import shutil
from collections import defaultdict
from pathlib import Path
import nltk
from nltk.corpus import stopwords
import numpy as np
from PIL import Image, ImageFile
import random

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
            if os.path.isfile(filepath / product / img) and img != "title.txt" and img != "description.txt":
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
        for img_path in img_paths:
            product_file_path_and_title.append([img_path, title, description])
    return product_file_path_and_title


if __name__ == "__main__":

    # params
    src_folder = Path("./adidas1600_train")
    # src_folder = Path("./adidas400_val")
    dst_folder = Path("./fashionIQ")

    copy_move_img_and_rename(Path(src_folder))
    img_path_list = src_folder_to_dict_of_file_path_and_title(src_folder)
    # print(img_path_list)
    cap_general_train_json = []
    for i in range(len(img_path_list)):
        for j in range(i + 1, len(img_path_list)):
            # get img name
            img_name_i = os.path.basename(img_path_list[i][0]).split('/')[-1]
            img_name_i = img_name_i.split('.')[0]
            img_name_j = os.path.basename(img_path_list[j][0]).split('/')[-1]
            img_name_j = img_name_j.split('.')[0]
            # print(img_name_i)
            if img_path_list[i][1] != img_path_list[j][1]:
                # randomly select pair with probability
                if random.random() < 0.1:
                    cap_general_train_json.append({"target": img_name_j, "candidate": img_name_i, "captions": ["change from " + img_path_list[i][1] + " to " + img_path_list[j][1], "change from " + img_path_list[j][1] + " to " + img_path_list[i][1]]})
                
    # print(cap_general_train_json)
    # dst_folder / "captions"
    # if file not exist, create file
    if not os.path.exists(dst_folder / "captions"):
        os.makedirs(dst_folder / "captions")
    if not os.path.exists(dst_folder / "captions" / "cap.general.train.json"):
        open(dst_folder / "captions" / "cap.general.train.json", "x").close()
    with open(dst_folder / "captions" / "cap.general.train.json", "w") as f:
        json.dump(cap_general_train_json, f)

    # dst_folder / "image_splits"
    split_general_train_json = []
    for i in range(len(img_path_list)):
        # get img name
        img_name_i = os.path.basename(img_path_list[i][0]).split('/')[-1]
        img_name_i = img_name_i.split('.')[0]
        split_general_train_json.append(img_name_i)
    if not os.path.exists(dst_folder / "image_splits"):
        os.makedirs(dst_folder / "image_splits")
    if not os.path.exists(dst_folder / "image_splits" / "split.general.train.json"):
        open(dst_folder / "image_splits" / "split.general.train.json", "x").close()
    with open(dst_folder / "image_splits" / "split.general.train.json", "w") as f:
        json.dump(split_general_train_json, f)
    
    # dst_folder / "images"
    if not os.path.exists(dst_folder / "images"):
        os.makedirs(dst_folder / "images")
    for img_title in img_path_list:
        img_name = os.path.basename(img_title[0]).split('/')[-1]
        # img_name = img_name.split('.')[0]
        shutil.copyfile(img_title[0], dst_folder / "images" / img_name)

    # dst_folder / "tags"
    asin2attr_general_train_json = []
    stop_words = set(stopwords.words('english'))
    for i in range(len(img_path_list)):
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
        
        asin2attr_general_train_json.append({img_name_i : description_tokens})
    if not os.path.exists(dst_folder / "tags"):
        os.makedirs(dst_folder / "tags")
    if not os.path.exists(dst_folder / "tags" / "asin2attr.general.train.json"):
        open(dst_folder / "tags" / "asin2attr.general.train.json", "x").close()
    with open(dst_folder / "tags" / "asin2attr.general.train.json", "w") as f:
        json.dump(asin2attr_general_train_json, f)