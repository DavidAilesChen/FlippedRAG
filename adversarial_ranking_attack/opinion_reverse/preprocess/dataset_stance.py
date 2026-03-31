import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(os.path.dirname(curdir))
sys.path.insert(0, prodir)
greatgrandparent_dir = os.path.dirname(os.path.dirname(prodir))

import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import random
import torch

class Stance_Dataset(object):
    def __init__(self, tokenizer, random_seed=666):
        self.tokenizer = tokenizer
        self.random_seed = random_seed
        self.tokenizer = tokenizer
        self.data_for_ft_path = '/opinion_pro_data/fnc/finetune_stances.pkl'
        self.stance_data_fnc_path = "/opinion_pro_data/fnc/train_stances_15docs_4polarity.pkl"
        self.detected_fnc_stances_path = "/opinion_pro_data/fnc/detected_stances.pkl"
        self.procon_data_path = greatgrandparent_dir+"/opinion_pro/procons_passages.pkl"

    def fnc_finetune_label_mapping(self, label):
        if label == "agree":
            return 1
        elif label == "disagree":
            return 0
        else:
            return 2
        
    def procon_label_mapping(self, label):
        if label.startswith("Pro"):
            return 1
        elif label.startswith("Con"):
            return 0
        else:
            return 2

    def load_fnc_stance(self, compress = True):
        with open(self.stance_data_fnc_path, "rb") as f:
            data = pkl.load(f)# dict:{Headline:[[Body ID,Stance], .....]}
            articles = pkl.load(f)
            if compress:
                for key in articles.keys():
                    articles[key] = articles[key].replace("  ", " ").replace("\n", "")
        return data, articles

    def load_fuc_finetune_data(self, compress = True, val_portion = 0.1):
        with open(self.data_for_ft_path, "rb") as f:
            data = pkl.load(f)
            articles = pkl.load(f)
            if compress:
                for key in articles.keys():
                    articles[key] = articles[key].replace("  ", " ").replace("\n", "")
        data_reset = []
        for key in data.keys():
            for row in data[key]:
                data_reset.append([(key, articles[row[0]]), self.fnc_finetune_label_mapping(row[1])])
        random.shuffle(data_reset)
        thred = int(len(data_reset)*val_portion)
        data_main = data_reset[thred:]
        data_dev = data_reset[:thred]
        return data_main, data_dev
    
    def load_detected_fnc_stance(self, compress = True):#Get oringin_data(including id), articles and processed data after stance detection.
        with open(self.detected_fnc_stances_path, "rb") as f:
            data = pkl.load(f)# dict:{Headline:[[Body ID,Stance], .....]}
            articles = pkl.load(f)
            if compress:
                for key in articles.keys():
                    articles[key] = articles[key].replace("  ", " ").replace("\n", "")
            queries_list = list(data.keys())
            data_process = {}
            for i in range(0,len(queries_list[:])):
                origin_stance_list = data[queries_list[i]]
                for j in range(len(origin_stance_list)):
                    origin_stance_list[j].extend([articles[origin_stance_list[j][0]], queries_list[i]])
                data_process[queries_list[i]] = origin_stance_list
        return data, articles, data_process

    def load_procon_data(self, num = 5):
        with open(self.procon_data_path, "rb") as f:
            data = pkl.load(f)
            data_process = {}
            for t in data.keys():
                argument_items = data[t]
                for i in range(len(argument_items)):
                    argument_items[i] = [i, self.procon_label_mapping(argument_items[i][0]), None, argument_items[i][2], t]
                data_process[t] = argument_items
        return data, data_process

        
