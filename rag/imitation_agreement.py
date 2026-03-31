import sys
import os
# from sarge import run

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

from tqdm import tqdm

import csv
import gzip
import codecs
import pandas as pd
import numpy as np
import os
import tarfile
import shutil
import zipfile
import collections
import pickle as pkl
import math
from scipy.stats import kendalltau


data_folder = prodir + '/data/msmarco_passage'
runs_folder = prodir + '/bert_ranker/results/runs/'
victim_label_triples = data_folder + '/pseudo_set/used_1_pairwise.pseudo.small.1280000.pkl'
simulator_label_triples = data_folder + '/pseudo_set/pairwise.pseudo.pseudo.small.1280000.pkl'

runs_folder = curdir + '/results/runs'

def inter_agreement_triples(pkl_1, pkl_2):
    with open(pkl_1, 'rb') as f:
        _ = pkl.load(f)
        labels_1 = pkl.load(f)
    
    with open(pkl_2, 'rb') as f:
        _ = pkl.load(f)
        labels_2 = pkl.load(f)

    a = np.array(labels_1)
    b = np.array(labels_2)

    p0 = 1. * (a == b).sum() / a.shape[0]
    a_true_percentage = 1. * a.sum() / a.shape[0]
    b_true_percentage = 1. * b.sum() / b.shape[0]
    pe = (a_true_percentage * b_true_percentage) + ((1. - a_true_percentage) * (1. - b_true_percentage))
    print("P0: %.2f, Pe = %.2f" % (p0, pe))
    return (p0 - pe) / (1.0 - pe)


def rbo_score(l1, l2, p, max_depth = 10):
    if not l1 or not l2:
        return 0
    s1 = set()
    s2 = set()
    score = 0.0
    max_depth = min(len(l1), len(l2))
    for d in range(max_depth):
        s1.add(l1[d])
        s2.add(l2[d])
        avg_overlap = len(s1 & s2) / (d + 1)
        score += math.pow(p, d) * avg_overlap
    return (1 - p) * score

def rank_overlap(l1, l2, p, max_depth = 10):
    if not l1 or not l2:
        return 0
    s1 = set()
    s2 = set()
    score = 0.0
    max_depth = min(len(l1), len(l2))
    for d in range(max_depth):
        s1.add(l1[d])
        s2.add(l2[d])
        avg_overlap = len(s1 & s2) / (d + 1)
        score += math.pow(p, d) * avg_overlap
    return score / max_depth

def rbo_dict_score(dic1, dic2, p , max_depth = 10):
    accumulate = []
    for q in dic1.keys():
        # truth_list = [t[0] for t in sorted(dic1[q].items(), key = lambda e:e[1])]
        truth_list = [t[0] for t in sorted(dic1[q].items(), key = lambda e:e[1], reverse=True)]
        pred_list = [t[0] for t in sorted(dic2[q].items(), key = lambda e:e[1], reverse=True)]
        # print(pred_list[:13])
        rbo = rbo_score(truth_list, pred_list, p = p, max_depth=max_depth)
        accumulate.append(rbo)
    print(accumulate)
    return np.mean(accumulate)

def load_runs(runs_path):
    runs = collections.defaultdict(list)
    with open(runs_path, 'r') as f:
        for line in f:
            qid, _, did, _, _, _ = line.strip().split('\t')
            # qid, did, _ = line.strip().split('\t')
            runs[qid].append(did)
    return runs


def top_n_overlap(runs_path_1, runs_path_2, topn=10):
    runs1 = load_runs(runs_path_1)
    runs2 = load_runs(runs_path_2)

    sim_ratio_list = []
    for qid, dids in runs1.items():
        target_dids = dids[:topn]
        another_dids = runs2[qid][:topn]
        tmp_sim_cnt = 0
        tmp_cnt = 0
        for did in target_dids:
            if did in another_dids:
                tmp_sim_cnt += 1
            tmp_cnt += 1
        sim_ratio_list.append(tmp_sim_cnt / (tmp_cnt + 0.0))
    print("Top@{} functional similarity inter: {}".format(topn, np.mean(sim_ratio_list)))

    return np.mean(sim_ratio_list)

def top_n_overlap_sim(l1, l2, topn=10):
    # print(len(l1), len(l2))
    max_depth = min(topn, len(l1), len(l2))
    # max_depth = topn
    tmp_sim_cnt = 0
    tmp_cnt = 0
    for d in range(max_depth):
        if l1[d] in l2[:max_depth]:
            tmp_sim_cnt+=1
        tmp_cnt+=1
    return tmp_sim_cnt / (tmp_cnt + 0.0)

def top_n_overlap_dic_sim(dic1, dic2, topn=10):
    accumulate = []
    for q in dic1.keys():
        truth_list = [t[0] for t in sorted(dic1[q].items(), key = lambda e:e[1], reverse=True)]
        pred_list = [t[0] for t in sorted(dic2[q].items(), key = lambda e:e[1], reverse=True)]
        overlap = top_n_overlap_sim(truth_list, pred_list, topn=topn)
        accumulate.append(overlap)
    return np.mean(accumulate)

def avg_rbo(runs_path_1, runs_path_2, topn=10, p=0.7):
    runs1 = load_runs(runs_path_1)
    runs2 = load_runs(runs_path_2)

    rbo_list = []
    for qid, dids in runs1.items():
        target_dids = dids[:topn]
        another_dids = runs2[qid][:topn]

        tmp_rbo = rbo_score(target_dids, another_dids, p=p)
        rbo_list.append(tmp_rbo)
    print("Top@{} functional similarity rbo: \t{}".format(topn, np.mean(rbo_list)))
    return np.mean(rbo_list)

def avg_tau(runs_path_1, runs_path_2, topn=10):
    runs1 = load_runs(runs_path_1)
    runs2 = load_runs(runs_path_2)

    tau_list = []
    for qid, dids in runs1.items():
        target_dids = dids[:topn]
        another_dids = runs2[qid][:topn]

        tmp_tau = kendalltau(target_dids, another_dids)
        tau_list.append(tmp_tau[0])
    
    print("Top@{} functional similarity: \t{}".format(topn, np.mean(tau_list)))


if __name__ == "__main__":
    dr_runs = '/msmarco/train/runs/runs.DR_coCondenser_all_msmarco.target_test.csv'
    rm_runs = '/msmarco/train/runs/runs.mini_all_batch20.suro_test.csv'
    p=0.7
    top_n_overlap(dr_runs, rm_runs, topn=10)
    print("######################################")
    top_n_overlap(dr_runs, rm_runs, topn=20)
    print('\n')

    avg_rbo(dr_runs, rm_runs, topn=10, p=p)
    print("######################################")
    avg_rbo(dr_runs, rm_runs, topn=1000, p=p)


