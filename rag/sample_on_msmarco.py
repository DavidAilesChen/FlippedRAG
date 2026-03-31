import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(os.path.dirname(curdir))
sys.path.insert(0, prodir)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from tqdm import tqdm

import csv
import gzip
import codecs
import pandas as pd
import os
import tarfile
import numpy as np
import zipfile
import collections
import pickle as pkl
import random
import faiss
from transformers import AutoTokenizer

from Condenser_model import CondenserForPairwiseModel_msmarco
from rag.dataset import RAG_Dataset
from condenser import sim_score
from bert_ranker_utils import accumulate_list_by_pid, accumulate_list_by_qid_2_dic, accumulate_list_by_qid, accumulate_list_by_qid_and_pid
from dense_retrieval_reranking import sim_ranker_for_set
from imitation_agreement import rbo_score,top_n_overlap_sim, rank_overlap
from test_between_LLM_RM import msmarco_run_bm25_load

current_file = __file__
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)
model_dir = os.path.dirname(os.path.dirname(grandparent_dir))+"/model_hub"

ms_data_folder = prodir + '/data/msmarco_passage'
sampled_triples_path = prodir + '/data/msmarco_passage/triples_from_runs'
runs_data_folder = prodir + '/bert_ranker/results/runs'
msmarco_collection_path = grandparent_dir+"/msmarco/msmarco_passage/collection_queries/collection.tsv"
msmarco_queries_path = grandparent_dir+'/msmarco/msmarco_passage/collection_queries/queries.dev.tsv'
run_bm25 = grandparent_dir+'/msmarco/msmarco_passage/sampled_set/run_bm25.tsv'

runs_ms_338 = runs_data_folder + '/runs.bert-base-uncased.pairwise.triples.Oct_20.victim.eval_full_dev1000.csv'
# runs_mono_bert_large = sampled_triples_path + '/BERT_Large_dev_run.tsv'
runs_bert_large = grandparent_dir+'/msmarco/train/runs/runs.DR_coCondenser_run_bm25_2_top1000.target_bm25_dot.csv'
run_condenser_5_up_bm25 = grandparent_dir+"/msmarco/train/runs/runs.DR_coCondenser_run_bm25_with_join_5_up_run_bm25.target_bm25_dot.csv"
run_contriever_bm25_origin = grandparent_dir+"/msmarco/train/runs_for_contriever/runs.DR_contriever-msmarco_on_bm25_origin.target_bm25_dot.csv"
run_ance_bm25_origin_nomessycode = grandparent_dir+"/msmarco/train/runs_for_ance/runs.DR_ance-msmarco_on_run_bm25_origin_no_messy_code.target_bm25_dot.csv"
runs_nbbert_on_5_up_bm25 = grandparent_dir+'/msmarco/train/runs/runs.NBbert_on_5_up_run_bm25_top600.suro_5_up_run_bm25_with_join.csv'
runs_nbbert_on_bm25_origin_collection = grandparent_dir+"/msmarco/train/runs_tempt/runs.NBbert_no_im_on_trecdl2019.suro_bm25_with_origin_collection_top2000.csv"
runs_nbbert_on_bm25_origin_nomessycode = grandparent_dir+"/msmarco/train/runs_for_ance/runs.NBbert_no_im_on_bm25_with_origin_no_messy_code_allq.suro_bm25.csv"
runs_distilbert_cat = runs_data_folder + '/runs.distilbert-cat-margin_mse-T2-msmarco.public.bert.msmarco.Tue_Nov_2.eval_full_dev1000.csv'
runs_MiniLM_L_12 = runs_data_folder + '/runs.ms-marco-MiniLM-L-12-v2.public.bert.msmarco.Thu_Jan_13.eval_full_dev1000.csv'

random_seed = 777

# ever used top 10 last 100
def sample_from_msmarco(run_path, save_pre_fix, top_n=25, last_sample=4, q_num = 100, measure = "dot"):
    """
    sample pairs from run_bm25.tsv in msmarco
    """
    #preparation
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    target_model = CondenserForPairwiseModel_msmarco.from_pretrained(model_dir+"/condenser/msmarco-bert-co-condensor")
    target_model.to('cuda')

    # load doc_id=pid to string
    print("LOADING colletion...")
    collection_df = pd.read_csv(msmarco_collection_path, sep='\t', names=['docid', 'document_string'])
    collection_df['docid'] = collection_df['docid'].astype(str)
    collection_str = collection_df.set_index('docid').to_dict()['document_string']
    # load query
    print("LOADING queries...")
    query_df = pd.read_csv(msmarco_queries_path, names=['qid','query_string'], sep='\t')
    query_df['qid'] = query_df['qid'].astype(str)
    queries_str = query_df.set_index('qid').to_dict()['query_string']
    print("QUERY_NUM:",len(queries_str))

    relevant_pairs_dict = collections.defaultdict(list)
    with open(run_bm25, 'r') as f:
        for line in tqdm(f):
            qid, did, query, passage, label = line.strip().split('\t')
            if qid == 'qid':
                continue
            # qid, did, _ = line.strip().split('\t')
            relevant_pairs_dict[qid].append(did)
    
    print("Store Q amount:", len(relevant_pairs_dict.keys()))
    
    sampled_triples_ids = []
    for seed, (qid, ori_did_list) in tqdm(enumerate(relevant_pairs_dict.items())):
        if len(ori_did_list)<=100:
            continue
        if seed >= q_num:
            break
        # for top n
        query = queries_str[str(qid)]
        passages = [collection_str[n] for n in ori_did_list]
        dis, ind = sim_ranker_for_set(target_model, query, passages, k=len(passages) ,measure=measure)
        score = dis[0]
        if measure == "dot":
            order_index = np.argsort(-score)
        elif measure == "L2":
            order_index = np.argsort(score)
        did_list = [ori_did_list[t] for t in order_index]
        tmp_top_n = top_n if len(did_list) > top_n else len(did_list)
        top_n_dids = did_list[:tmp_top_n]
        last_dids = did_list[tmp_top_n:]
        for i in range(tmp_top_n):
            random.seed(random_seed + seed + i)
            pos_did = top_n_dids[i]
            for j in range(i + 1, tmp_top_n):
                neg_did = top_n_dids[j]
                sampled_triples_ids.append((qid, pos_did, neg_did))
            # for top n corresponding random negative sampling from another 990
            if len(last_dids) < last_sample:
                selected_last_neg = last_dids
            else:
                selected_last_neg = random.sample(last_dids, last_sample)
            for tmp_did in selected_last_neg:
                sampled_triples_ids.append((qid, pos_did, tmp_did))
    
    # save qid pos_did neg_did list
    print("Sampled {} triples from: {}".format(len(sampled_triples_ids), run_path))
    # triples_ids_df = pd.DataFrame(sampled_triples_ids, columns=["qid", "pos_did", "neg_did"])
    # save_ids_path = save_pre_fix + '_ids.top_{}_last_{}.csv'.format(top_n, last_sample)
    # triples_ids_df.to_csv(save_ids_path, sep='\t', index=False, header=True)

    sampled_triples_text_dict = {}
    for (qid, pos_did, neg_did) in sampled_triples_ids:
        if qid not in sampled_triples_text_dict:
            sampled_triples_text_dict[qid] = [[collection_str[pos_did], collection_str[neg_did]]]
        else:
            sampled_triples_text_dict[qid].append([collection_str[pos_did], collection_str[neg_did]])
    
    # final_text_triples_df = pd.DataFrame(sampled_triples_text_list)
    # save_text_path = save_pre_fix + '_text.top_{}_last_{}.csv'.format(top_n, last_sample)
    # final_text_triples_df.to_csv(save_text_path, sep='\t', index=False, header=False)
    # save_to_pkl
    pkl_path = grandparent_dir+"/msmarco/ranks/msmarco_run_bm25_cocondenser_dot_sample_1000q_triple.pkl"
    with open(pkl_path, "wb") as f:
        pkl.dump(sampled_triples_text_dict, f)
    f.close()
    print(pkl_path," SAVED!")

def sample_negative_from_runs(run_path, pos_path, top_n=3, last_sample=60, q_num = 100, measure = "dot", last_bound = -1, repeat_times = 10):
    if top_n > last_bound and last_bound != -1:
        raise ValueError()
    from LocalEmbedding import localEmbedding, localEmbedding_sentence, localEmbedding_contriever, localEmbedding_ance, localEmbedding_QWEN3
    CON_NAME = model_dir+'/msmarco-roberta-base-ance-firstp'
    CON_NAME = model_dir+'/dpr'
    CON_NAME = model_dir+'/Qwen3-Embedding-4B'
    device = 'cuda'
    embedding_model = localEmbedding_QWEN3(
            CON_NAME,
            device
    )

    #Load data
    print("LOADING colletion...")
    # collection_df = pd.read_csv(msmarco_collection_path, sep='\t', names=['docid', 'document_string'])
    # collection_df['docid'] = collection_df['docid'].astype(str)
    # collection_str = collection_df.set_index('docid').to_dict()['document_string']
    # print("LOADING queries...")
    # query_df = pd.read_csv(msmarco_queries_path, names=['qid','query_string'], sep='\t')
    # query_df['qid'] = query_df['qid'].astype(str)
    # queries_str = query_df.set_index('qid').to_dict()['query_string']
    data, collection_str, queries_str = msmarco_run_bm25_load()

    relevant_pairs_dict = collections.defaultdict(list)
    with open(run_path, 'r') as f:
        print("LOADING runs: ", run_path)
        for line in f:
            qid, _, did, _, _, _ = line.strip().split('\t')
            # qid, did, _ = line.strip().split('\t')
            relevant_pairs_dict[qid].append(did)
    f.close()

    with open(pos_path, "rb") as f_1:
        print("LOADING pos: ", pos_path)
        data_pos = pkl.load(f_1)
    f_1.close()

    data_new = {}
    bad_num = 0
    qids = list(data_pos.keys())
    q_num = 0
    for qid in tqdm(qids[0:500]):
        q_num+=1
        print("QID:", qid)
        query = queries_str[str(qid)]
        info = data_pos[qid]
        pos_pid = list((info.keys()))
        sample_target = relevant_pairs_dict[qid][0:last_bound]
        top_truth = relevant_pairs_dict[qid][0:top_n+7]
        # rbo = rbo_score(pos_pid, top_truth, 0.7, top_n)
        overlap = rank_overlap(pos_pid, top_truth, p=1, max_depth=3)
        print("OVERLAP:", overlap, "|", pos_pid, top_truth)
        sample_target_2 = list(set(sample_target)-set(pos_pid))
        print(sample_target_2)
        data_new[qid] = []
        for i in range(len(pos_pid)):
            random.seed(random_seed + q_num +  i + 1)
            # pos_text = info[pos_pid[i]]
            pos_text = info[pos_pid[i]][1]
            for j in range(i + 1, len(pos_pid)):
                neg_text = info[pos_pid[j]][1]
                if top_n < 10:

                    # dis_pos = sim_score(embedding_model, query, pos_text, measure=measure)
                    # dis_neg = sim_score(embedding_model, query, neg_text, measure=measure)
                    # if measure == "dot":
                    #     dis_gap = dis_pos - dis_neg
                    # else:
                    #     dis_gap = dis_neg - dis_pos
                    # if dis_gap <= 0:
                    #     print("L2_SCORE:", dis_pos, " with neg ", j,"-pid:",pos_pid[j]," ", dis_neg, " So BAD!")
                    #     bad_num +=(1*repeat_times)
                    #     print(bad_num)
                    # else:
                    #     print("L2_SCORE:", dis_pos, " with neg ", j,"-pid:",pos_pid[j]," ", dis_neg, " GOOD!")

                    data_new[qid].extend([[pos_text, neg_text]]*repeat_times)
                    # print(data_new[qid])
                else:
                    data_new[qid].append([pos_text, neg_text])
            neg_dids = random.sample(sample_target_2, last_sample)
            neg_texts = [collection_str[t] for t in neg_dids]
            for j in range(last_sample):
                # query = template_no_data
                data_new[qid].append([pos_text, neg_texts[j]])

                # dis_pos = sim_score(embedding_model, query, pos_text, measure=measure)
                # dis_neg = sim_score(embedding_model, query, neg_texts[j], measure=measure)
                # if measure == "dot":
                #     dis_gap = dis_pos - dis_neg
                # else:
                #     dis_gap = dis_neg - dis_pos
                # if dis_gap <= 0:
                #     print("L2_SCORE:", dis_pos, " with neg ", sample_target_2.index(neg_dids[j]),"-pid:",neg_dids[j]," ", dis_neg, " So BAD!")
                #     bad_num +=1
                #     print(bad_num)
                # else:
                #     print("L2_SCORE:", dis_pos, " with neg ", sample_target_2.index(neg_dids[j]),"-pid:",neg_dids[j]," ", dis_neg, " GOOD!")

        print("DATA:",len(data_new[qid]))
    
    print("BAD:", bad_num)
    pkl_path = grandparent_dir+"/msmarco/ranks/extract_from_llm_QWEN/rag_500q_random_sample_in_QWEN3x10-50fromnbrank_top60_FAST.pkl"
    # with open(pkl_path, "wb") as f:
    #     pkl.dump(data_new, f)
    # f.close()
    # print(pkl_path," SAVED!")
            

def sample_from_runs(run_path, save_pre_fix, top_n=3, last_sample=60, q_num = 100, measure = "dot", last_bound = -1, repeat_times = 5):
    if top_n > last_bound and last_bound != -1:
        raise ValueError()
    #laod data
    relevant_pairs_dict = collections.defaultdict(list)
    with open(run_path, 'r') as f:
        for line in f:
            qid, _, did, _, _, _ = line.strip().split('\t')
            # qid, did, _ = line.strip().split('\t')
            relevant_pairs_dict[qid].append(did)
    
    sampled_triples_ids = []
    for seed, (qid, did_list) in enumerate(relevant_pairs_dict.items()):
        # for top n
        tmp_top_n = top_n if len(did_list) > top_n else len(did_list)
        top_n_dids = did_list[:tmp_top_n]
        if last_bound != -1:
            # print("BOUND:", tmp_top_n, "--", last_bound)
            last_dids = did_list[tmp_top_n:last_bound]
        else:
            last_dids = did_list[tmp_top_n:]
        for i in range(tmp_top_n):
            random.seed(random_seed + seed + i)
            pos_did = top_n_dids[i]
            for j in range(i + 1, tmp_top_n):
                neg_did = top_n_dids[j]
                if top_n < 10:
                    sampled_triples_ids.extend([(qid, pos_did, neg_did)]*repeat_times)
                else:
                    sampled_triples_ids.append((qid, pos_did, neg_did))
            # for top n corresponding random negative sampling from another 990
            if len(last_dids) < last_sample:
                selected_last_neg = last_dids
            else:
                selected_last_neg = random.sample(last_dids, last_sample)
            # print([did_list.index(t) for t in selected_last_neg])
            for tmp_did in selected_last_neg:
                sampled_triples_ids.append((qid, pos_did, tmp_did))
            # print(len(sampled_triples_ids))
    
    # save qid pos_did neg_did list
    print("Sampled {} triples from: {}".format(len(sampled_triples_ids), run_path))
    # triples_ids_df = pd.DataFrame(sampled_triples_ids, columns=["qid", "pos_did", "neg_did"])
    # save_ids_path = save_pre_fix + '_ids.top_{}_last_{}.csv'.format(top_n, last_sample)
    # triples_ids_df.to_csv(save_ids_path, sep='\t', index=False, header=True)
    # print("Saved sampled triples ids into : {}".format(save_ids_path))

    # load doc_id to string
    # print("LOADING colletion...")
    # collection_df = pd.read_csv(msmarco_collection_path, sep='\t', names=['docid', 'document_string'])
    # collection_df['docid'] = collection_df['docid'].astype(str)
    # collection_str = collection_df.set_index('docid').to_dict()['document_string']

    # # load query
    # print("LOADING queries...")
    # query_df = pd.read_csv(msmarco_queries_path, names=['qid','query_string'], sep='\t')
    # query_df['qid'] = query_df['qid'].astype(str)
    # queries_str = query_df.set_index('qid').to_dict()['query_string']

    _, collection_str, queries_str = msmarco_run_bm25_load()

    #LOAD MODEL
    # target_model = CondenserForPairwiseModel_msmarco.from_pretrained("/data_share/model_hub/condenser/msmarco-bert-co-condensor")
    # target_model.to('cuda')
    #test
    sampled_triples_text_list = []
    sampled_triples_text_dict = {}
    p=[0.01,0.99]
    for (qid, pos_did, neg_did) in tqdm(sampled_triples_ids):
    #     if random.choices([True, False], weights = p):
    #         pos_sim = sim_score(target_model, queries_str[qid], collection_str[pos_did])
    #         neg_sim = sim_score(target_model, queries_str[qid], collection_str[neg_did])
    #         if measure == "dot":
    #             if pos_sim<=neg_sim:
    #                 raise ValueError("Similarity Wrong!")
    #             else:
    #                 print("OK")
    #                 # pass
        # sampled_triples_text_list.append((queries_str[qid], collection_str[pos_did], collection_str[neg_did]))
        if qid not in sampled_triples_text_dict:
            sampled_triples_text_dict[qid] = [[collection_str[pos_did], collection_str[neg_did]]]
        else:
            sampled_triples_text_dict[qid].append([collection_str[pos_did], collection_str[neg_did]])
    for q in sampled_triples_text_dict:
        print(len(sampled_triples_text_dict[q]))
    
    # final_text_triples_df = pd.DataFrame(sampled_triples_text_list)
    # save_text_path = save_pre_fix + '_text.top_{}_last_{}.csv'.format(top_n, last_sample)
    # final_text_triples_df.to_csv(save_text_path, sep='\t', index=False, header=False)
    # print("Saved sampled triples text into : {}".format(save_text_path))

    pkl_path = grandparent_dir+"/msmarco/ranks/msmarco_run_bm25_origin_nomessycode_ance_dot_top3x10-50sample_top100_allq_triple.pkl"#msmarco_run_bm25_cocondenser_dot_top10-20sample_top100_allq_triple.pkl
    with open(pkl_path, "wb") as f:
        pkl.dump(sampled_triples_text_dict, f)
    f.close()
    print(pkl_path," SAVED!")

def sample_pos_neg_split(run_path_pos, run_path_neg, top_n=3, last_sample=60, measure = "dot", last_bound = -1, repeat_times = 5, q_limit = 400):
    if top_n > last_bound and last_bound != -1:
        raise ValueError()
    #load data
    relevant_pairs_dict_pos = collections.defaultdict(list)
    with open(run_path_pos, 'r') as f:
        for line in f:
            qid, _, did, _, _, _ = line.strip().split('\t')
            # qid, did, _ = line.strip().split('\t')
            relevant_pairs_dict_pos[qid].append(did)

    relevant_pairs_dict_neg = collections.defaultdict(list)
    with open(run_path_neg, 'r') as f_2:
        for line in f_2:
            qid, _, did, _, _, _ = line.strip().split('\t')
            # qid, did, _ = line.strip().split('\t')
            relevant_pairs_dict_neg[qid].append(did)
    
    sampled_triples_ids = []
    q_num = 0
    for seed, (qid, did_list) in enumerate(relevant_pairs_dict_neg.items()):
        q_num+=1
        if q_num>q_limit:
            break
        # for top n
        tmp_top_n = top_n if len(did_list) > top_n else len(did_list)
        did_list_pos = relevant_pairs_dict_pos[qid]
        top_n_dids = did_list_pos[:tmp_top_n]
        neg_did_list = relevant_pairs_dict_neg[qid]
        if last_bound != -1:
            print("BOUND:", tmp_top_n, "--", last_bound)
            last_dids = neg_did_list[0:last_bound]
        else:
            last_dids = neg_did_list[0:]
        
        for id in top_n_dids:
            if id in last_dids:
                last_dids.remove(id)

        for i in range(tmp_top_n):
            random.seed(random_seed + seed + i)
            pos_did = top_n_dids[i]
            for j in range(i + 1, tmp_top_n):
                neg_did = top_n_dids[j]
                if top_n < 10:
                    sampled_triples_ids.extend([(qid, pos_did, neg_did)]*repeat_times)
                else:
                    sampled_triples_ids.append((qid, pos_did, neg_did))
            # for top n corresponding random negative sampling from another 990
            if len(last_dids) < last_sample:
                selected_last_neg = last_dids
            else:
                selected_last_neg = random.sample(last_dids, last_sample)
            # print([did_list.index(t) for t in selected_last_neg])
            for tmp_did in selected_last_neg:
                sampled_triples_ids.append((qid, pos_did, tmp_did))
            # print(len(sampled_triples_ids))
    
    # save qid pos_did neg_did list
    print("Sampled {} triples from: {} and {}".format(len(sampled_triples_ids), run_path_pos, run_path_neg))
    # triples_ids_df = pd.DataFrame(sampled_triples_ids, columns=["qid", "pos_did", "neg_did"])
    # save_ids_path = save_pre_fix + '_ids.top_{}_last_{}.csv'.format(top_n, last_sample)
    # triples_ids_df.to_csv(save_ids_path, sep='\t', index=False, header=True)
    # print("Saved sampled triples ids into : {}".format(save_ids_path))

    # load doc_id to string
    # print("LOADING colletion...")
    # collection_df = pd.read_csv(msmarco_collection_path, sep='\t', names=['docid', 'document_string'])
    # collection_df['docid'] = collection_df['docid'].astype(str)
    # collection_str = collection_df.set_index('docid').to_dict()['document_string']

    # # load query
    # print("LOADING queries...")
    # query_df = pd.read_csv(msmarco_queries_path, names=['qid','query_string'], sep='\t')
    # query_df['qid'] = query_df['qid'].astype(str)
    # queries_str = query_df.set_index('qid').to_dict()['query_string']
    # print("QUERY_NUM:",len(queries_str))

    _, collection_str, queries_str = msmarco_run_bm25_load()

    #LOAD MODEL
    # target_model = CondenserForPairwiseModel_msmarco.from_pretrained("/data_share/model_hub/condenser/msmarco-bert-co-condensor")
    # target_model.to('cuda')
    #test
    sampled_triples_text_list = []
    sampled_triples_text_dict = {}
    p=[0.01,0.99]
    for (qid, pos_did, neg_did) in tqdm(sampled_triples_ids):
    #     if random.choices([True, False], weights = p):
    #         pos_sim = sim_score(target_model, queries_str[qid], collection_str[pos_did])
    #         neg_sim = sim_score(target_model, queries_str[qid], collection_str[neg_did])
    #         if measure == "dot":
    #             if pos_sim<=neg_sim:
    #                 raise ValueError("Similarity Wrong!")
    #             else:
    #                 print("OK")
    #                 # pass
        # sampled_triples_text_list.append((queries_str[qid], collection_str[pos_did], collection_str[neg_did]))
        if qid not in sampled_triples_text_dict:
            sampled_triples_text_dict[qid] = [[collection_str[pos_did], collection_str[neg_did]]]
        else:
            sampled_triples_text_dict[qid].append([collection_str[pos_did], collection_str[neg_did]])
    for q in tqdm(sampled_triples_text_dict):
        print(len(sampled_triples_text_dict[q]))
    
    # final_text_triples_df = pd.DataFrame(sampled_triples_text_list)
    # save_text_path = save_pre_fix + '_text.top_{}_last_{}.csv'.format(top_n, last_sample)
    # final_text_triples_df.to_csv(save_text_path, sep='\t', index=False, header=False)
    # print("Saved sampled triples text into : {}".format(save_text_path))

    pkl_path = grandparent_dir+"/msmarco/ranks/msmarco_run_bm25_origin_nomessycode_dpr_dot_top3x10-50sample_fromnbrank_top60_500q_triple_FAST.pkl"#msmarco_run_bm25_cocondenser_dot_top10-20sample_top100_allq_triple.pkl
    with open(pkl_path, "wb") as f:
        pkl.dump(sampled_triples_text_dict, f)
    f.close()
    print(pkl_path," SAVED!")

if __name__ == "__main__":
    # save_triples_prefix = sampled_triples_path + '/minilm_l12_sampled_triples'
    # sample_from_dev_runs(runs_MiniLM_L_12, save_triples_prefix)
    #save_triples_prefix = sampled_triples_path + '/bert_large_sampled_triples'
    # sample_from_msmarco(runs_bert_large, save_triples_prefix)
    # sample_from_runs(run_ance_bm25_origin_nomessycode, save_triples_prefix, top_n= 3, last_bound=100, last_sample=50, repeat_times=10)
    #sample_negative_from_runs(runs_nbbert_on_bm25_origin_nomessycode, "/mnt/data_share/chenzhuo/msmarco/ranks/extract_from_llm_ance/pos.pkl", top_n=3, last_sample=50, last_bound=60, repeat_times=10)
    # path_pos_dpr=grandparent_dir+'/msmarco/train/runs_for_dpr/runs.DR_dpr-msmarco_on_run_bm25_origin_no_messy_code_2.target_bm25_dot.csv'
    
    #sample_pos_neg_split(path_pos_dpr, runs_nbbert_on_bm25_origin_nomessycode, top_n= 3, last_bound=60, last_sample=50, repeat_times=10, q_limit=500)
    run_path=grandparent_dir+'/msmarco/train/runs/runs.nbbert_only.suro_bm25.csv'
    pos_path=grandparent_dir+'/msmarco/ranks/extract_from_llm_QWEN/pos.pkl'
    sample_negative_from_runs(run_path, pos_path,top_n= 3, last_bound=60, last_sample=50, repeat_times=10)
