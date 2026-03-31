import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tqdm import tqdm
import pandas as pd
import pickle as pkl
import json
import argparse
import random
from collections import defaultdict
from dataset import RAG_Dataset
from test_between_LLM_RM import msmarco_load_and_sample

current_file = __file__
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)

def get_msmarco(size = 10, num = 1):
    collection_path = grandparent_dir+'/msmarco/msmarco_passage/collection_queries/collection.tsv'
    queries_path = grandparent_dir+'/msmarco/msmarco_passage/collection_queries/queries.dev.tsv'
    qrels_path = grandparent_dir+'/msmarco/msmarco_passage/qrels.dev.tsv'
    # load doc_id=pid to string
    collection_df = pd.read_csv(collection_path, sep='\t', names=['docid', 'document_string'])
    collection_df['docid'] = collection_df['docid'].astype(str)
    collection_str = collection_df.set_index('docid').to_dict()['document_string']
    # load query
    query_df = pd.read_csv(queries_path, names=['qid','query_string'], sep='\t')
    query_df['qid'] = query_df['qid'].astype(str)
    queries_str = query_df.set_index('qid').to_dict()['query_string']
    print(len(queries_str))
    # load qrels
    qrels_df = pd.read_csv(qrels_path, delim_whitespace= True,names=['qid', 'iter', 'docid', 'relevance'])
    
    temt = ''
    terms = []
    nums = 0
    for i in range(len(qrels_df)):
        if qrels_df.loc[i, 'qid'] == temt:
            nums+=1
        else:
            if nums>=3 and temt not in terms:
                terms.append(temt)
                print(temt,":", nums)
            temt = qrels_df.loc[i, 'qid']
            nums = 1
    print("T",len(terms))
    """"""
    qid_str = '1102375'
    target_q = [[qid_str, queries_str[qid_str]]]
    print(target_q)
    """
    for i in range(num):
        qid = list(queries_str.keys())[i]
        target_q.append([qid, queries_str[qid]])
    """
    relevant_num = int(size/2)
    passages = []
    #
    for t in target_q:
        print("qid:", t[0])
        for i in range(len(qrels_df)):
            if i in [3,67, 89, 104, 177]:
                passages.append([qrels_df.loc[i, 'docid'], 0, collection_str[str(qrels_df.loc[i, 'docid'])]])
            if str(qrels_df.loc[i, 'qid']) == t[0] and qrels_df.loc[i, 'relevance'] > 0:
                passages.append([qrels_df.loc[i, 'docid'], qrels_df.loc[i, 'relevance'], collection_str[str(qrels_df.loc[i, 'docid'])]])
            if len(passages)>=size:
                break
    return target_q, passages, [t[2] for t in passages]

def load_text_from_pkl(data_path):
    texts = []
    with open(data_path, 'rb') as f:
        data = pkl.load(f)
    for head in data.keys():
        text_list = data[head]
        for line in text_list:
            texts.append(line[2])
    return texts, data

def procon_label_mapping(label):
        if label.startswith("Pro"):
            return 1
        elif label.startswith("Con"):
            return 0
        else:
            return 2

def load_procon_data(path, num = 5):
        with open(path, "rb") as f:
            data = pkl.load(f)
            data_process = {}
            for t in data.keys():
                argument_items = data[t]
                for i in range(len(argument_items)):
                    argument_items[i] = [i, procon_label_mapping(argument_items[i][0]), None, argument_items[i][2], t]
                data_process[t] = argument_items
        f.close()
        return data, data_process

def read_triggers():
    """
    input: {query: [id, label, none, passage, query]}
    Return: {query: [id, trigger, passage]}
    """
    path =grandparent_dir+"/opinion_pro/triggers/"+"pat_16-45_zero_passages_contriever_from_nb_ep4_dropout_blackbox_contriever_bm25_origin_sample3x10-50fromnbrank_top60_dot_400q_batch128_tripledev_4e5.pkl"
    with open(path, 'rb') as f:
        data = pkl.load(f)
    f.close()
    text_dict = {}#{query:text_dic}
    for q in data.keys():
        sub_data = [[t[0], t[2], t[3]] for t in data[q]]#
        # label_rank = [t[1] for t in data[q]]
        text_dic = {t[3]:t[1] for t in data[q]}#text：label
        text_dict[q] = text_dic
        data[q] = sub_data
    return data, text_dict

def create_rank_train():
    # from pipline_RAG import rag_generation
    object_data = RAG_Dataset("34")
    data = object_data.load_msmarco_data_with_pro_con_labels_for_rag(sample_num=4)
    object_data.save_to_pkl(grandparent_dir+"/msmarco/samples/4_up.pkl", data)

def create_imitate_data():
    object_data = RAG_Dataset("34")
    # object_data.load_data_to_ranking()
    object_data.load_data_to_generate()

def test_trec_dl():
    object_data = RAG_Dataset("34")
    object_data.trec_dl_test()

def merge_data(path_1, path_2):
    with open(path_1, "rb") as f1:
         data_1 = pkl.load(f1)
    f1.close()
    data_1_amount = 0
    data_2_amount = 0
    for q in data_1:
        data_1_amount += len(data_1[q])
    print("FIRST DATA AMOUNT:", data_1_amount, len(data_1.keys()))
    data_2, pid_2_text, qid_2_text = msmarco_load_and_sample()
    
    sample_num = 10
    print("DATA_2 KEYS:", len(data_2.keys()))
    sample_key_num = 30
    sample_keys = []
    for q in data_2.keys():
        if q not in data_1:
            sample_keys.append(q)
        if len(sample_keys) >= sample_key_num:
            break
    
    for q in sample_keys:
        sub_data = data_2[q]
        poses = [pid_2_text[t[0]] for t in sub_data if t[1]>0]
        negs = [pid_2_text[t[0]] for t in sub_data if t[1]==0]
        data_1[q] = []
        for j in range(sample_num):
            data_1[q].append([random.sample(poses, 1)[0], random.sample(negs, 1)[0]])
    
    for q in data_1:
        if q in data_2:
            sub_data = data_2[q]
            poses = [pid_2_text[t[0]] for t in sub_data if t[1]>0]
            negs = [pid_2_text[t[0]] for t in sub_data if t[1]==0]
            for j in range(sample_num):
                data_1[q].append([random.sample(poses, 1)[0], random.sample(negs, 1)[0]])
            print(qid_2_text[q])
        else:
            print("do not exist!")
    
    for q in data_1:
        data_2_amount += len(data_1[q])
    print(("SECOND DATA AMOUNT:", data_2_amount, len(data_1.keys())))
    
    with open(path_2, "wb") as f2:
        pkl.dump(data_1, f2)
        print("SAVED IN ",path_2)
    f2.close()


if __name__ == '__main__':
    # get_msmarco()
    # create_rank_train()
    # create_imitate_data()
    # test_trec_dl()
    merge_data(grandparent_dir+"/msmarco/ranks/msmarco_run_bm25_sample_1000_generation_2.pkl", grandparent_dir+"/msmarco/ranks/msmarco_run_bm25_sample_q1-q65_generation+msmarco_600.pkl")#amount:605