import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
import torch
from torch import cuda
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
# from apex import amp
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    BertForNextSentencePrediction,
)
from adv_ir.attack_methods import pairwise_anchor_trigger
from sklearn.metrics import ndcg_score
import pickle as pkl

device = 'cuda' if cuda.is_available() else 'cpu'

def topk_proportion(original_label_rank, later_label_rank, args, topk = [3, 6]):
    polarity = args.polarity
    result = {
        
    }
    for k in topk:
        target = original_label_rank[:k]
        score = len([t for t in target if t == polarity])/k
        print("In original top"+str(k)+" stance "+str(polarity)+" proportion：", score)
        target_2 = later_label_rank[:k]
        score_2 = len([t for t in target_2 if t == polarity])/k
        print("In manipulated top"+str(k)+" stance "+str(polarity)+"proportion：", score_2)
        result['before-top'+str(k)] = score
        result["later-top"+str(k)] = score_2
        print("Boost：", score_2-score)
        result["top"+str(k)+" boost"] = score_2-score
    return result

def topk_proportion_to_length(original_label_rank, later_label_rank, args, topk):
    polarity = args.polarity
    result = {
        
    }
    for k in topk:
        target = original_label_rank[:k]
        score = len([t for t in target if t == polarity])/k
        print("in original top1/2 stance "+str(polarity)+" proportion：", score)
        target_2 = later_label_rank[:k]
        score_2 = len([t for t in target_2 if t == polarity])/k
        print("In manipulated top1/2 stance "+str(polarity)+" proportion：", score_2)
        result['before-top1/2'] = score
        result["later-top1/2"+str(k)] = score_2
        print("Boost：", score_2-score)
        result["top1/2 boost"] = score_2-score
    return result

def topk_mutual_score(original_label_rank, later_label_rank, args, topk = [3, 6]):
    polarity = args.polarity
    result = {
        
    }
    for k in topk:
        if k == 1/2:
            k = int(len(original_label_rank)/2)
            name = "1/2"
        else:
            name = k
        target = original_label_rank[:k]
        score = len([t for t in target if t != polarity])/k
        print("In original top"+str(k)+" not stance "+str(polarity)+" proportion：", score)
        target_2 = later_label_rank[:k]
        score_2 = len([t for t in target_2 if t != polarity])/k
        print("In manipulated top"+str(k)+" not stance "+str(polarity)+" proportion：", score_2)
        if score == 0:
            score_2 = 1
            score = 1
        result["later-top-not-proportion"+str(name)] = score_2
        print("top"+str(name)+"mutual boosted score：", 1-(score_2/score))
        result["top"+str(name)+"mutual boosted score"] = 1-(score_2/score)

    return result

def recall_score(original_label_rank, later_label_rank, args):
    polarity = args.polarity
    result = {
        
    }
    target_num = len([t for t in original_label_rank if t == polarity])
    target = original_label_rank[:target_num]
    score = len([t for t in target if t == polarity])/target_num
    print("In original top"+str(target_num)+" stance "+str(polarity)+" proportion：", score)
    target_2 = later_label_rank[:target_num]
    score_2 = len([t for t in target_2 if t == polarity])/target_num
    print("In manipulated top"+str(target_num)+" stance "+str(polarity)+" proportion：", score_2)
    if score == 0:
        score_2 = 1
        score = 1
    print("top"+str(target_num)+"recall boosted score：", score_2-score)
    result["recall boosted score"] = score_2-score
    return result

def avg_rank_boost(original_label_rank, later_label_rank, args):
    origin_rank_sum = sum([t for t in range(len(original_label_rank)) if original_label_rank[t] == args.polarity])
    later_rank_sum = sum([t for t in range(len(later_label_rank)) if later_label_rank[t] == args.polarity])
    boost_sum = origin_rank_sum - later_rank_sum
    # print([original_label_rank.index(t) for t in original_label_rank if t == args.polarity],"%%", [later_label_rank.index(t) for t in later_label_rank if t == args.polarity])
    avg_boost_rank = boost_sum/len([t for t in original_label_rank if t == args.polarity])
    return avg_boost_rank, boost_sum, len([t for t in original_label_rank if t == args.polarity])

def relabel_polarity(tag, labels):
    for t in range(len(labels)):
        if labels[t] == tag:
            labels[t] = 3
        elif labels[t] == 2:
            labels[t] = 1
        else:
            labels[t] = 0
    return labels

def cal_NDCG(scores, labels, k=10):
    scores = np.array([scores])
    labels = np.array([labels])
    ndcg = ndcg_score(labels, scores, k = k)
    return ndcg

def save_plus_to_pkl(path_1, path_2, final_path):
        """
        data_dict:{}
        """
        with open(path_1, "rb") as f:
            data_1 = pkl.load(f)
            # if "Is a College Education Worth It?" in data_1:
            #     print("hello!")
            #     data_1.pop("Is a College Education Worth It?")
            print("1 Already has: ", len(data_1.keys()))
        f.close()
        with open(path_2, "rb") as f:
            data = pkl.load(f)
            print("2 Already has: ", len(data.keys()))
            data.update(data_1)
        f.close()
        print("Now we has: ", len(data.keys()))
        with open(final_path, "wb") as f_2:
            pkl.dump(data, f_2)
        f_2.close()
        print(final_path," ADD!")

def rbo_score(l1, l2, p, max_depth = 10):
    # if not l1 or not l2:
    #     return 0
    s1 = set()
    s2 = set()
    score = 0.0
    max_depth = min(max_depth, len(l1), len(l2))
    for d in range(max_depth):
        s1.add(l1[d])
        s2.add(l2[d])
        avg_overlap = len(s1 & s2) / (d + 1)
        score += math.pow(p, d) * avg_overlap
    return (1 - p) * score

if __name__ == "__main__":
    trigger_dir = '/opinion_pro_data/procon_trigger/'
    final_path = trigger_dir+'education_pat_passages_mini_im48_test.pkl'
    save_plus_to_pkl(trigger_dir+'pat_3+6+13+69_zero_education_passages_mini_im48.pkl', trigger_dir+'pat_32+39_one_education_passages_mini_im48.pkl',final_path)
    # save_plus_to_pkl(trigger_dir+'pat_40_zero_passages_mini_im.pkl', final_path, final_path)
    # save_plus_to_pkl(trigger_dir+'pat_43+44+52_zero_passages_mini_im.pkl', final_path, final_path)
    # save_plus_to_pkl(trigger_dir+'pat_61_zero_passages_mini_im.pkl', final_path, final_path)
    # save_plus_to_pkl(trigger_dir+'pat_54_zero_passages_mini_im.pkl', final_path, final_path)