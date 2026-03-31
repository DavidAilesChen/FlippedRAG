from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import LLMChain,HuggingFacePipeline,PromptTemplate
import os
import pickle as pkl
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
# from LocalEmbedding import localEmbedding
from scipy.stats import spearmanr
import random
import re
import json

current_file = __file__
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)

def extract_by_symbol(text, symbol = "<( )>", segment = "\n\n"):
    if symbol == "[[ ]]":
        pattern = r'\[\[.*?\]\]'
        try:
            match = re.search(pattern, str(text), re.DOTALL)
            l = re.split("\[\[|\]\]", match.group(0))
            string = l[1]
            print("Extracted content：", string)
        except:
            string = "\n\n"
    elif symbol == "<( )>":
        pattern = r'<\(.*?\)>'
        try:
            match = re.search(pattern, str(text), re.DOTALL)
            l = re.split("<\(|\)>", match.group(0))
            string = l[1]
        except AttributeError:
            string = "\n\n"

    string_list = string.split(segment)
    
    return string_list

def interaction_data_input(long_text, patterns, topk = 3):
    text_list = []
    # topk = len(patterns)
    #topk = int(input("How many passages are there?"))
    for i in range(topk):
        tag = "f"
        while tag == "f" or tag == "F":
            choose, match = quick_match(long_text, patterns[i])
            if choose:
                reminder_3 = "No "+str(i)+" passage is[["+match+"]]?"
                tag = input(reminder_3)
                if tag == "f" or tag == "F":
                    input_str = input("So it is?")
                    tag = "t"
                else:
                    input_str = match
            else:
                reminder_1 = "No "+str(i)+" passage is?"
                input_str = input(reminder_1)
                reminder_2 = "IS <<"+str(input_str)+">>?"
                tag = input(reminder_2)
        text_list.append(input_str)
        print("#####################")
    
    return text_list

def interaction_data_input(long_text, patterns, topk=3):
    text_list = []
    
    for i in range(topk):
        choose, match = quick_match(long_text, patterns[i])
        if choose:
            text_list.append(match)
        else:
            return None
    
    return text_list


def save_to_pkl(path, data_dict):
        """
        data_dict:{}
        """
        with open(path, "wb") as f:
            pkl.dump(data_dict, f)
        f.close()
        print(path," SAVED!")
    
def save_plus_to_pkl(path, data_dict):
    """
    data_dict:{}
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pkl.load(f)
            print("Already has: ", len(data.keys()))
            data.update(data_dict)
        f.close()
        print("Now we has: ", len(data.keys()))
        with open(path, "wb") as f_2:
            pkl.dump(data, f_2)
        f_2.close()
        print(path," ADD!")
    else:
        save_to_pkl(path, data_dict)

def load_from_pkl(path):
        with open(path, "rb") as f:
            print("OPEN ", path)
            data = pkl.load(f)
        f.close()
        return data

def quick_match(long, sub):
    if sub in long:
        return True, sub
    else:
        return False, None

def print_out(input):
    print(input)
    return input
    
def transition(input):
    return input
    
def merge_one_pkl_to_target(target_path, add_path):
    data_add = load_from_pkl(add_path)
    print("ADD DATA:", len(data_add))
    save_plus_to_pkl(target_path, data_add)

if __name__ == '__main__':
    merge_one_pkl_to_target(grandparent_dir+"msmarco/ranks/extract_from_llm_QWEN/rag_generate_pos_500q_random_sample_in_QWEN3x10-50fromrandomsample_top50.pkl", grandparent_dir+"msmarco/ranks/extract_from_llm_QWEN/rag_generate_pos_600_896_200.pkl")

