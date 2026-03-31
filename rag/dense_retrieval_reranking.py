import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(os.path.dirname(curdir))
sys.path.insert(0, prodir)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import random
import torch
import faiss
from transformers import AutoTokenizer

from Condenser_model import CondenserForPairwiseModel_msmarco
from rag.dataset import RAG_Dataset
from condenser import sim_score
from bert_ranker_utils import accumulate_list_by_pid, accumulate_list_by_qid_2_dic, accumulate_list_by_qid, accumulate_list_by_qid_and_pid

current_file = __file__
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)
model_dir = os.path.dirname(os.path.dirname(grandparent_dir))+"/model_hub"

msmarco_collection_path = grandparent_dir+"/msmarco/msmarco_passage/collection_queries/collection.tsv"
msmarco_queries_path = grandparent_dir+'/msmarco/msmarco_passage/collection_queries/queries.dev.tsv'
msmarco_qrels_path = grandparent_dir+'/msmarco/msmarco_passage/qrels.dev.tsv'
run_bm25 = grandparent_dir+'/msmarco/msmarco_passage/sampled_set/run_bm25.tsv'

def trec_dl_load(self, sample_num = 30,):
    pkl_path = grandparent_dir+"/opinion_pro/trec_dl_2019/trec_dl2019_passage_test1000_full.pkl"
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            print("Loading instances from {}".format(pkl_path))
            examples = pkl.load(f)#[query, passage]
            labels = pkl.load(f)
            qids = pkl.load(f)
            pids = pkl.load(f)
    else:
        raise ValueError("{} not exists".format(pkl_path))
    
    return examples, labels, qids, pids

def batch_tokenize(examples, labels, qids, pids, tokenizer, batch_size = 64, max_seq_len=256,):
    for i in tqdm(range(0, len(examples), batch_size), desc='Processing:'):
        tmp_examples = examples[i: i+batch_size]
        tmp_qids = qids[i: i+batch_size]
        tmp_pids = pids[i: i+batch_size]
        tmp_labels = torch.tensor(labels[i: i + batch_size], dtype=torch.long)
        batch_encoding_query = tokenizer([e[0] for e in tmp_examples], max_length=max_seq_len,padding="max_length", truncation=True, return_tensors='pt')#query
        batch_encoding_passage = tokenizer([e[1] for e in tmp_examples], 
                                    max_length=max_seq_len,padding="max_length", truncation=True, return_tensors='pt')
        yield batch_encoding_query, batch_encoding_passage, tmp_examples, tmp_qids, tmp_pids, tmp_labels

def sim_score_(model, query, passage, measure = "dot"):
    query_emb = model.encode(query)
    passage_emb = model.encode(passage)
    d = query_emb.shape[1]
    if measure == 'L2':
        index = faiss.IndexFlatL2(d)
    elif measure == 'dot':
        index = faiss.IndexFlatIP(d)
    index.add(np.array([passage_emb]))
    dis, ind = index.search(np.array(query_emb), 1)
    distance = dis[0][0]
    return distance

def sim_ranker(model, query, candidates, k=10, measure = "dot"):
    """
    INPUT:
    dense model;str;list
    """
    query_emb = model.encode_(query).cpu().numpy()
    doc_embs = model.encode_(candidates).cpu().numpy()
    d = query_emb.shape[1]

    dis_score = []
    for i in range(query_emb.shape[0]):
        # index = faiss.IndexFlatL2(d) 
        if measure == 'L2':
            index = faiss.IndexFlatL2(d)
        elif measure == 'dot':
            index = faiss.IndexFlatIP(d)

        index.add(np.array([doc_embs[i]])) 
        dis, ind = index.search(np.array([query_emb[i]]), 1)
        dis_score.append(dis[0][0])

    # for i in range(len(dis_score)):
    #      dis_score[ind[0][i]] = dis[0][i]
    
    return [dis_score], ind

def sim_ranker_for_set(model, query, candidates, k=10, measure = "dot"):
    """
    INPUT:
    dense model;str;list
    """
    query_emb = model.encode(query)[0].cpu().numpy()
    doc_embs = model.encode(candidates).cpu().numpy()
    d = len(query_emb)
    query_emb = np.array([query_emb])
    if measure == 'L2':
        index = faiss.IndexFlatL2(d)
    elif measure == 'dot':
        index = faiss.IndexFlatIP(d)  
    index.add(doc_embs)
    dis, ind = index.search(query_emb, k)
    dis_score = [0]*len(dis[0])
    for i in range(len(dis_score)):
         dis_score[ind[0][i]] = dis[0][i]
    
    return np.array([dis_score]), ind

def dense_ranking(measure = 'L2' ,device = 'cuda'):
    #preparation
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    target_model = CondenserForPairwiseModel_msmarco.from_pretrained(model_dir+"/condenser/msmarco-bert-co-condensor")
    target_model.to('cuda')
    all_labels = []
    all_qids = []
    all_pids = []
    all_examples = []
    all_logits = []

    data_class = RAG_Dataset(tokenizer=tokenizer)

    for batch_encoding_q, batch_encoding_p, batch_encoding, tmp_labels, tmp_qids, tmp_pids, tmp_examples in data_class.data_generator_ranking_dev_for_dr(
                                                                                batch_size=64,
                                                                                max_seq_len=512):
        true_labels = tmp_labels.to(device)
        batch_encoding_q = batch_encoding_q.to(device)
        batch_encoding_p = batch_encoding_p.to(device)
        if measure == 'dot':
            outputs_target = target_model(
                query = batch_encoding_q,
                pos = batch_encoding_p,
                neg = batch_encoding_p
                #labels=true_labels
            )
            logits = outputs_target[0].cpu()
        elif measure == 'L2':
            outputs_target, _ = sim_ranker(target_model, batch_encoding_q, batch_encoding_p, k=len(tmp_labels))
            logits = outputs_target[0]
        
        all_logits += logits
        all_qids += tmp_qids
        all_pids += tmp_pids
        all_examples += tmp_examples
        all_labels += tmp_labels

    #Aggregate and Rank
    pid_2_text = accumulate_list_by_pid(all_examples, all_pids)
    qid_2_text = accumulate_list_by_qid_2_dic(all_examples, all_qids)

    query_logits = accumulate_list_by_qid_and_pid(all_logits, all_pids, all_qids)
    query_labels = accumulate_list_by_qid_and_pid(all_labels, all_pids ,all_qids)
    #construct_labels
    new_examples = []
    new_labels = []

    for q in query_logits.keys():
        print("#####################")
        print("Query:", qid_2_text[q])
        logits_list = query_logits[q]
        true_labels = query_labels[q]
        if measure == "L2":
            sorted_logits = sorted(logits_list.items(), key=lambda x: x[1], reverse=False)
        elif measure == "dot":
            sorted_logits = sorted(logits_list.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(sorted_logits)):
            pid = sorted_logits[i][0]
            label =  int(query_labels[q][pid])
            sorted_logits[i] = (pid, sorted_logits[i][1], label)
        print("Sorted Logits:", sorted_logits[:30], "LEN: ", len(sorted_logits))
        sorted_truth = sorted(true_labels.items(), key=lambda x: x[1], reverse=True)
        print("Sorted_labels:", sorted_truth[:20])

    all_logits = all_logits
    all_logits_target, _ = accumulate_list_by_qid(all_logits, all_qids)
    all_pids, all_qids = accumulate_list_by_qid(all_pids, all_qids)

    mode = 'target_test'
    output_dir = grandparent_dir+'/msmarco/train/'
    runs_list = []
    run_id = "DR_coCondenser_dot_flat"
    for scores, qids, pids in zip(all_logits_target, all_qids, all_pids):
        sorted_idx = np.array(scores).argsort()[::-1]
        sorted_scores = np.array(scores)[sorted_idx]
        sorted_qids = np.array(qids)[sorted_idx]
        sorted_pids = np.array(pids)[sorted_idx]
        for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
                runs_list.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'BERT-Pair'))
        runs_df = pd.DataFrame(runs_list, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
        runs_df.to_csv(output_dir + '/runs/runs.' + run_id + '.' + mode + '.csv', sep='\t', index=False, header=False)
    print("DONE!")

def dense_L2(device = 'cuda', num = 10, pseudo_value = 2):
    #preparation
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    target_model = CondenserForPairwiseModel_msmarco.from_pretrained(model_dir+"/condenser/msmarco-bert-co-condensor")
    target_model.to('cuda')
    all_labels = []
    all_qids = []
    all_pids = []
    all_examples = []
    all_logits = []

    data_class = RAG_Dataset(tokenizer=tokenizer)

    for batch_encoding_q, batch_encoding_p, batch_encoding, tmp_labels, tmp_qids, tmp_pids, tmp_examples in data_class.data_generator_ranking_dev_for_dr(
                                                                                batch_size=64,
                                                                                max_seq_len=512):
        all_qids += tmp_qids
        all_pids += tmp_pids
        all_examples += tmp_examples
        all_labels += tmp_labels
    
    pid_2_text = accumulate_list_by_pid(all_examples, all_pids)
    qid_2_text = accumulate_list_by_qid_2_dic(all_examples, all_qids)
    query_labels = accumulate_list_by_qid_and_pid(all_labels, all_pids ,all_qids)
    agg_examples, _ = accumulate_list_by_qid(all_examples, all_qids)
    # agg_examples = accumulate_list_by_qid_and_pid(all_examples, all_pids, all_qids)
    pids_l, qids_l = accumulate_list_by_qid(all_pids, all_qids)
    logits_l = []
    for l in tqdm(agg_examples):
        query = l[0][0]
        passages = [t[1] for t in l]
        dis, ind = sim_ranker_for_set(target_model, query, passages, k=len(passages), measure="L2")
        logits_l.append(dis[0])
    logit_dic = {}
    for i in range(len(qids_l)):
        pids = pids_l[i]
        qids = qids_l[i]
        logits = logits_l[i]
        logit_dic[qids[0]] = {}
        for j in range(len(pids)):
            logit_dic[qids[0]][pids[j]] = logits[j]

    #construct newdata
    new_examples = []
    new_labels = []
    new_pids = []
    new_qids = []
    for q in tqdm(query_labels.keys()):
        logits_list = logit_dic[q]
        true_labels = query_labels[q]
        sorted_logits = sorted(logits_list.items(), key=lambda x: x[1], reverse=False)
        for i in range(len(sorted_logits)):
            pid = sorted_logits[i][0]
            label =  int(query_labels[q][pid])
            sorted_logits[i] = (pid, sorted_logits[i][1], label)
        print("######################")
        print("Sorted Logits:", sorted_logits[:30], "LEN: ", len(sorted_logits))
        # sorted_truth = sorted(true_labels.items(), key=lambda x: x[1], reverse=True)
        relevent_pids = [t for t in true_labels.keys() if true_labels[t]>0]
        num = len(relevent_pids)
        for j in range(len(sorted_logits)):
            if j<num: #and sorted_logits[j][0] not in relevent_pids
                new_examples.append([qid_2_text[q],pid_2_text[sorted_logits[j][0]]])
                new_labels.append(pseudo_value)
                new_pids.append(sorted_logits[j][0])
                new_qids.append(q)
                print("@@:", sorted_logits[j][0], "--", true_labels[sorted_logits[j][0]])
            else:
                new_examples.append([qid_2_text[q],pid_2_text[sorted_logits[j][0]]])
                # new_labels.append(true_labels[sorted_logits[j][0]])
                new_labels.append(0)
                new_pids.append(sorted_logits[j][0])
                new_qids.append(q)
    
    print(len(new_examples), len(new_labels), len(new_qids), len(new_pids))

    #save_to_pkl
    pkl_path = grandparent_dir+"/opinion_pro/trec_dl_2019/trec_dl2019_passage_test1000_full_orderby_cocondenser_only.pkl"
    with open(pkl_path, "wb") as f:
        pkl.dump(new_examples, f)
        pkl.dump(new_labels, f)
        pkl.dump(new_qids, f)
        pkl.dump(new_pids, f)
    f.close()
    print(pkl_path," SAVED!")

def DR_on_msmarco(q_num = 50, sample_times = 20, measure = "dot"):
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
    print(len(queries_str))
    # load qrels
    print("LOADING qrefs...")
    # qrels_df = pd.read_csv(msmarco_qrels_path, delim_whitespace= True,names=['qid', 'iter', 'docid', 'relevance'])
    
    # qrels_df = pd.read_csv(run_bm25, sep='/t', names=['qid', 'docid', 'query', 'passage',  'relevance'])
    import csv
    qrels_df = []
    with open(run_bm25, "r") as f:
        csv_reader = csv.reader(f)
        for line in tqdm(csv_reader):
            line = ''.join(line)
            line = line.split('\t')
            if len(line) != 5:
                print(line)
            qrels_df.append(line)

    #data_input
    data = {}
    for i in tqdm(range(1, len(qrels_df))):
        if qrels_df[i][0] not in data and len(data.keys()) < q_num:
            data[qrels_df[i][0]] = {qrels_df[i][1]:[int(qrels_df[i][4]), collection_str[qrels_df[i][1]]]}
        elif qrels_df[i][0] in data:
            data[qrels_df[i][0]][qrels_df[i][1]] = [int(qrels_df[i][4]), collection_str[str(qrels_df[i][1])]]
    
    print("Store Q amount:", len(data.keys()))
    
    new_data = {}
    for t in tqdm(data.keys()):
        new_data[t] = []
        query = queries_str[str(t)]
        passages = [data[t][n][1] for n in data[t].keys()]
        dis, ind = sim_ranker_for_set(target_model, query, passages, k=len(passages))
        score = dis[0]
        if measure == "dot":
            order_index = np.argsort(-score)
        elif measure == "L2":
            order_index = np.argsort(score)
        for j in range(sample_times):
            dual_idx = sorted(random.sample(range(1,100), k=2))
            if score[order_index[dual_idx[0]]] < score[order_index[dual_idx[1]]] and measure == "dot":
                print("BIG PROBLEM!", dual_idx, score[order_index.index(dual_idx[0])], score[order_index.index(dual_idx[1])])

            # pos_dis = L2_score(target_model, query, passages[dual_idx[0]])
            pos_dis = sim_score(target_model, query, passages[order_index[dual_idx[0]]])
            # neg_dis = L2_score(target_model, query, passages[dual_idx[1]])
            neg_dis = sim_score(target_model, query, passages[order_index[dual_idx[1]]])
            # dis, ind = L2_ranker_for_set(target_model, query, [passages[order_index[dual_idx[0]]], passages[order_index[dual_idx[1]]]], k=len(passages))
            # pos_dis, neg_dis = dis[0][0], dis[0][1]
            if pos_dis <= neg_dis:
                print("In Rank:", score[order_index[dual_idx[0]]], " To ", score[order_index[dual_idx[1]]])
                print("In SOlo:", pos_dis, " To ", neg_dis)

            # new_data[t].append([passages[dual_idx[0]], passages[dual_idx[1]]])
            new_data[t].append([passages[order_index[dual_idx[0]]], passages[order_index[dual_idx[1]]]])

    # save_to_pkl
    pkl_path = grandparent_dir+"/msmarco/ranks/msmarco_run_bm25_cocondenser_dot_sample_1000_triple.pkl"
    with open(pkl_path, "wb") as f:
        pkl.dump(new_data, f)
    f.close()
    print(pkl_path," SAVED!")


if __name__ == '__main__':
    # dense_ranking(measure='L2')
    # dense_L2()
    DR_on_msmarco()


