# Import required packages
import numpy as np
import os
import pandas as pd
import argparse
import multiprocessing as mp
from multiprocessing import Pool
import json
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help = "path to the json file")
    parser.add_argument("--output_dir", help = "path to save the clean decsriptions")

    args = parser.parse_args()
    return args

"""This function loads the csv data"""
def load_data():
    args = get_args()
    json_dir = args.json
    json_data = json.load(open(json_dir))
    data = pd.DataFrame(json_data)
    return data

"""This function splits the data into train and test data"""
def split_data():
    data = load_data()
    images, captions = data["subfigure_file"], data['caption']      #data["object-and-view"]
    train_imgs, test_imgs, train_caps, test_caps = train_test_split(images, captions, random_state = 0, test_size = 0.01)
    return train_imgs, test_imgs, train_caps, test_caps


"""put the image names and description in a dictionary"""
def image_desc(images, captions):
    description = {}
    for img, caption in zip(images, captions):
        if img not in description:
            description[img] = caption
    
    return description

"""This function clean the text captions"""
def cleanDescription(images, captions):
    description = image_desc(images, captions)
    clean_description = {}
    for img, text in description.items():
        text = text.replace(";", " ")
        text = text.replace(".", "")
        text = text.strip()
        text = text.rstrip(".;")
        text = text.lower()
        clean_description[img] = text
    return clean_description

if __name__ == '__main__':
    args = get_args()
    output_dir = args.output_dir
    train_imgs, test_imgs, train_caps, test_caps = split_data()
    train_descriptions = cleanDescription(train_imgs, train_caps)
    test_descriptions = cleanDescription(test_imgs, test_caps)
    with open(os.path.join(output_dir, "train_descriptions.json"), 'w') as fp:
        json.dump(train_descriptions, fp)
        fp.close()

    with open(os.path.join(output_dir, "test_descriptions.json"), 'w') as f:
        json.dump(test_descriptions, f)
        f.close()
