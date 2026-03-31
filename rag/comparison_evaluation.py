from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import LLMChain,HuggingFacePipeline,PromptTemplate
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import pickle as pkl
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
from LocalEmbedding import localEmbedding,  localEmbedding_sentence
from scipy.stats import spearmanr
from imitation_agreement import top_n_overlap_sim, rbo_score
# from evaluate import cal_NDCG
# from metrics import evaluate_and_aggregate
import label_smoothing
import collections
import random
import re
import numpy as np
from rag.dataset import RAG_Dataset
from imitation_agreement import top_n_overlap_dic_sim, rbo_dict_score, avg_rbo, top_n_overlap
from condenser import sim_ranker, sim_score_for_passage_list
from bert_ranker_utils import accumulate_list_by_qid_and_pid, accumulate_list_by_pid, accumulate_list_by_qid_2_dic, accumulate_list_by_qid
from test_between_LLM_RM import msmarco_run_bm25_load

current_file = __file__
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)
model_dir = os.path.dirname(os.path.dirname(grandparent_dir))+"/model_hub"

msmarco_collection_path = grandparent_dir+'/msmarco/msmarco_passage/collection_queries/collection.tsv'
msmarco_queries_path = grandparent_dir+'/msmarco/msmarco_passage/collection_queries/queries.dev.tsv'
msmarco_qrels_path = grandparent_dir+'/msmarco/msmarco_passage/qrels.dev.tsv'
imitation_data_path = grandparent_dir+'/msmarco/samples/4_up.pkl'

def match_and_rank(data_new, label_new, model, topk=20):
    #{qid:{pid: text}}
    all_logits = []
    all_pids = []
    all_qids = []
    all_labels = []
    for qid in tqdm(data_new.keys()):
        all = list(data_new[qid].values())
        texts = [t[1] for t in all]
        queries = [t[0] for t in all]
        query = queries[0]
        db =  FAISS.from_texts(texts, model)
        search_result = db.similarity_search(query, k=topk, )
        search_result = [t.page_content for t in search_result]
        for i in range(len(search_result)):
            all_logits.append(len(search_result)-i)
            all_pids.append(list(data_new[qid].keys())[texts.index(search_result[i])])
            all_qids.append(qid)
            all_labels.append(label_new[qid][list(data_new[qid].keys())[texts.index(search_result[i])]])
    all_labels, _ = accumulate_list_by_qid(all_labels, all_qids)
    logit_pid_qid_dict_local = accumulate_list_by_qid_and_pid(all_logits, all_pids, all_qids)
    all_logits, _ = accumulate_list_by_qid(all_logits, all_qids)
    all_pids, all_qids = accumulate_list_by_qid(all_pids, all_qids)
    return all_logits, all_pids, all_qids, all_labels, logit_pid_qid_dict_local

def read_runs(path, organize = "dict"):
    if organize == "dict":
        data = {}
        with open(path, "r") as f:
            for line in f:
                qid, _, did, _, logit, _ = line.strip().split('\t')
                if qid not in data:
                    data[qid] = {did: float(logit)}
                else:
                    data[qid][did] = float(logit)
        f.close()
        return data
    else:
        raise ValueError("Not supported yet!")    

def test_between_dr_and_surrogate(loss_function = "label-smoothing-cross-entropy", smoothing=0.1, output_dir=grandparent_dir+"/msmarco/train/"):
    # LOAD model
    device = 'cuda'
    from Condenser_model import CondenserForPairwiseModel, CondenserForPairwiseModel_msmarco
    from Imitation_Encoder import MiniForPairwiseClassfy, NBBERTForPairwiseClassfy
    target_model = CondenserForPairwiseModel_msmarco.from_pretrained(grandparent_dir+"/msmarco-bert-co-condensor")
    target_model.eval()
    target_model.to(device)
    suro_model_path = grandparent_dir+"/nbbert_epoch80_runbm25_cocondensersample_dot_batch32.pt"
    suro_model = NBBERTForPairwiseClassfy.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
    suro_model.load_state_dict(torch.load(suro_model_path))
    suro_model.eval()
    suro_model.to(device)

    topk = 10
    #embedding
    BGE_NAME = model_dir+"/bge-large-en-v1.5"
    CON_NAME = model_dir+'/msmarco-bert-co-condensor'
    model_kwargs = {'device': 'cuda',
                    }
    encode_kwargs = {'normalize_embeddings': True}

    device = 'cuda'
    embedding_model = localEmbedding_sentence(
            CON_NAME,
            device
        )
    print("LOADED!!!")

    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")

    all_logits_target = []
    all_logits_suro = []
    all_labels = []
    all_qids = []
    all_pids = []
    all_examples = []
    cnt = 0
    
    mode = 'test'
    data_class = RAG_Dataset(tokenizer=tokenizer)
        
    for batch_encoding_q, batch_encoding_p, batch_encoding, tmp_labels, tmp_qids, tmp_pids, tmp_examples in data_class.data_generator_ranking_dev_for_dr(mode=mode,
                                                                                batch_size=32,
                                                                                max_seq_len=512):
        cnt += 1
        batch_encoding = batch_encoding.to(device)
        pos_input_ids = batch_encoding['input_ids'].to(device)
        pos_token_type_ids = batch_encoding['token_type_ids'].to(device)
        pos_attention_mask = batch_encoding['attention_mask'].to(device)
        neg_input_ids = batch_encoding['input_ids'].to(device)
        neg_token_type_ids = batch_encoding['token_type_ids'].to(device)
        neg_attention_mask = batch_encoding['attention_mask'].to(device)
        true_labels = tmp_labels.to(device)
        batch_encoding_q = batch_encoding_q.to(device)
        batch_encoding_p = batch_encoding_p.to(device)
        outputs_target = target_model(
                query = batch_encoding_q,
                pos = batch_encoding_p,
                neg = batch_encoding_p
                #labels=true_labels
        )
        # outputs_target = L2_ranker(target_model, batch_encoding_q, batch_encoding_p)


        outputs_suro = suro_model(
            input_ids_pos=pos_input_ids,
            attention_mask_pos=pos_attention_mask,
            token_type_ids_pos=pos_token_type_ids,
            input_ids_neg=neg_input_ids,
            attention_mask_neg=neg_attention_mask,
            token_type_ids_neg=neg_token_type_ids,
        )
        if loss_function == "label-smoothing-cross-entropy":
            loss_func = label_smoothing.LabelSmoothingCrossEntropy(smoothing)
        else:
            loss_func = nn.CrossEntropyLoss(size_average=False, reduce=True) 
        if outputs_suro[0].shape[1] >= 2:
            pass
            # val_loss = loss_func(outputs_suro[0].view(-1, 2), true_labels.view(-1))
            # if mode in ['dev']:
            #     self.writer_train.add_scalar('val_loss', val_loss, step)
            # elif mode in ['test']:
            #     pass

        logits_suro = outputs_suro[0]
        sim_logit = outputs_target[0]
        all_labels += true_labels.int().tolist() # this is required because of the weak supervision
        if len(outputs_target[0].shape)>1 and outputs_target[0].shape[1] >= 2:
            all_logits_target += sim_logit[:, 1].tolist()
        else:
            all_logits_target += sim_logit[:].tolist()
        if outputs_suro[0].shape[1] >= 2:
            all_logits_suro += logits_suro[:, 1].tolist()
        else:
            all_logits_suro += logits_suro[:, 0].tolist()
        # all_softmax_logits += torch.softmax(logits, dim=1)[:, 1].tolist()
        all_qids += tmp_qids
        all_pids += tmp_pids
        all_examples+= tmp_examples

        # if self.num_validation_batches!=-1 and cnt > self.num_validation_batches and mode == 'dev':
        #         break

    # data_new = accumulate_list_by_qid_and_pid(all_examples, all_pids ,all_qids)
    # label_new = accumulate_list_by_qid_and_pid(all_labels, all_pids ,all_qids)
    #accumulates per query
    all_labels, _ = accumulate_list_by_qid(all_labels, all_qids)
    # print(all_labels[:10])
    logit_pid_qid_dict_target = accumulate_list_by_qid_and_pid(all_logits_target, all_pids, all_qids)
    logit_pid_qid_dict_suro = accumulate_list_by_qid_and_pid(all_logits_suro, all_pids, all_qids)
    all_logits_target, _ = accumulate_list_by_qid(all_logits_target, all_qids)
    all_logits_suro, _ = accumulate_list_by_qid(all_logits_suro, all_qids)
    # all_softmax_logits, _ = accumulate_list_by_qid(all_softmax_logits, all_qids)
    all_pids, all_qids = accumulate_list_by_qid(all_pids, all_qids)

    #EXTRA
    # all_logits_local, all_pids_local, all_qids_local, all_labels_local, logit_pid_qid_dict_local = match_and_rank(data_new, label_new ,embedding_model, topk=20)

    res_dr = evaluate_and_aggregate(all_logits_target, all_labels, ['ndcg_cut_10', 'map', 'recip_rank'])
    # set_recall_rate = set_recall(self.data_class.truth_ranking ,logit_pid_qid_dict)
    rbo = rbo_dict_score(logit_pid_qid_dict_target, logit_pid_qid_dict_suro, p=0.7, max_depth=10)
    overlap = top_n_overlap_dic_sim(logit_pid_qid_dict_target, logit_pid_qid_dict_suro, topn=10)
    res_dr['inter'] = overlap
    res_dr['rbo'] = rbo

    for metric, v in res_dr.items():
        print("\n{} {} : {:3f}".format(mode, metric, v))
    
    # Save ranking results
    if mode in ['dev', 'test']:
        # mode = 'target_test_L2'
        # runs_list = []
        # run_id = "DR_coCondenser_all_msmarco"
        # for scores, qids, pids in zip(all_logits_target, all_qids, all_pids):
        #     sorted_idx = np.array(scores).argsort()[::-1]
        #     sorted_scores = np.array(scores)[sorted_idx]
        #     sorted_qids = np.array(qids)[sorted_idx]
        #     sorted_pids = np.array(pids)[sorted_idx]
        #     for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
        #         runs_list.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'BERT-Pair'))
        # runs_df = pd.DataFrame(runs_list, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
        # runs_df.to_csv(output_dir + '/runs/runs.' + run_id + '.' + mode + '.csv', sep='\t', index=False, header=False)

        # mode_suro = 'suro_test'
        # runs_list_suro = []
        # run_id_suro = "epoch80_runbm25_cocondensersample_dot"
        # for scores, qids, pids in zip(all_logits_suro, all_qids, all_pids):
        #     sorted_idx = np.array(scores).argsort()[::-1]
        #     sorted_scores = np.array(scores)[sorted_idx]
        #     sorted_qids = np.array(qids)[sorted_idx]
        #     sorted_pids = np.array(pids)[sorted_idx]
        #     for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
        #         runs_list_suro.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'BERT-Pair'))
        # runs_df_suro = pd.DataFrame(runs_list_suro, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
        # runs_df_suro.to_csv(output_dir + '/runs/runs.' + run_id_suro + '.' + mode_suro + '.csv', sep='\t', index=False, header=False)
        pass

def imitation_model_ranking(loss_function = "label-smoothing-cross-entropy", smoothing=0.1, output_dir=grandparent_dir+"/msmarco/train/", mode = 'test'):#ON TREC DL or BM25
    """
    mode:   test: ranking on trec_dl
            5_up_run_bm25_with_join: ranking on run_bm25 data
    """
    # mode = "5_up_run_bm25_with_join"
    # LOAD model
    device = 'cuda'
    from Imitation_Encoder import MiniForPairwiseClassfy, NBBERTForPairwiseClassfy, BertForPairwiseLearning, MinitForPairwiseLearning
    suro_model_path = grandparent_dir+"/msmarco/train/models_for_QWEN/NBbert_epoch4_dropout_QWEN_black_bm25_origin_nomessycode_sample3x10-50fromnbrank_top60_dot_500q_batch256_tripledev_4e5.pt"#mini_epoch24_batch32.pt
    suro_model = BertForPairwiseLearning.from_pretrained(model_dir+"/nboost_pt-bert-base-uncased-msmarco")#nboost/pt-bert-base-uncased-msmarco
    suro_model.load_state_dict(torch.load(suro_model_path))
    print("LOADED ", suro_model_path)
    suro_model.eval()
    suro_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir+"/nboost_pt-bert-base-uncased-msmarco")

    all_logits_suro = []
    all_labels = []
    all_qids = []
    all_pids = []
    all_examples = []
    cnt = 0
    
    data_class = RAG_Dataset(tokenizer=tokenizer)
        
    for batch_encoding_q, batch_encoding_p, batch_encoding, tmp_labels, tmp_qids, tmp_pids, tmp_examples in data_class.data_generator_ranking_dev_for_dr(mode=mode,
                                                                                batch_size=32,
                                                                                max_seq_len=512):#on trec_dl_2019
        cnt += 1
        batch_encoding = batch_encoding.to(device)
        pos_input_ids = batch_encoding['input_ids'].to(device)
        pos_token_type_ids = batch_encoding['token_type_ids'].to(device)
        pos_attention_mask = batch_encoding['attention_mask'].to(device)
        neg_input_ids = batch_encoding['input_ids'].to(device)
        neg_token_type_ids = batch_encoding['token_type_ids'].to(device)
        neg_attention_mask = batch_encoding['attention_mask'].to(device)
        true_labels = tmp_labels.to(device)
        # outputs_target = L2_ranker(target_model, batch_encoding_q, batch_encoding_p)

        outputs_suro = suro_model(
            input_ids_pos=pos_input_ids,
            attention_mask_pos=pos_attention_mask,
            token_type_ids_pos=pos_token_type_ids,
            input_ids_neg=neg_input_ids,
            attention_mask_neg=neg_attention_mask,
            token_type_ids_neg=neg_token_type_ids,
        )
        
        logits_suro = outputs_suro[0]
        all_labels += true_labels.int().tolist() # this is required because of the weak supervision
        if outputs_suro[0].shape[1] >= 2:
            all_logits_suro += logits_suro[:, 1].tolist()
        else:
            all_logits_suro += logits_suro[:, 0].tolist()
        # all_softmax_logits += torch.softmax(logits, dim=1)[:, 1].tolist()
        all_qids += tmp_qids
        all_pids += tmp_pids
        all_examples+= tmp_examples

        # if self.num_validation_batches!=-1 and cnt > self.num_validation_batches and mode == 'dev':
        #         break

    # data_new = accumulate_list_by_qid_and_pid(all_examples, all_pids ,all_qids)
    # label_new = accumulate_list_by_qid_and_pid(all_labels, all_pids ,all_qids)
    #accumulates per query
    all_labels, _ = accumulate_list_by_qid(all_labels, all_qids)
    logit_pid_qid_dict_suro = accumulate_list_by_qid_and_pid(all_logits_suro, all_pids, all_qids)
    all_logits_suro, _ = accumulate_list_by_qid(all_logits_suro, all_qids)
    # all_softmax_logits, _ = accumulate_list_by_qid(all_softmax_logits, all_qids)
    all_pids, all_qids = accumulate_list_by_qid(all_pids, all_qids)

    #EXTRA
    # all_logits_local, all_pids_local, all_qids_local, all_labels_local, logit_pid_qid_dict_local = match_and_rank(data_new, label_new ,embedding_model, topk=20)
    
    # Save ranking results
    if mode in ['dev', 'test', '5_up_run_bm25_with_join', 'bm25']:
    #     # mode = 'target_test_L2'
    #     # runs_list = []
    #     # run_id = "DR_coCondenser_all_msmarco"
    #     # for scores, qids, pids in zip(all_logits_target, all_qids, all_pids):
    #     #     sorted_idx = np.array(scores).argsort()[::-1]
    #     #     sorted_scores = np.array(scores)[sorted_idx]
    #     #     sorted_qids = np.array(qids)[sorted_idx]
    #     #     sorted_pids = np.array(pids)[sorted_idx]
    #     #     for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
    #     #         runs_list.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'BERT-Pair'))
    #     # runs_df = pd.DataFrame(runs_list, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
    #     # runs_df.to_csv(output_dir + '/runs/runs.' + run_id + '.' + mode + '.csv', sep='\t', index=False, header=False)

        mode_suro = 'suro_'+mode#suro_5_up_run_bm25_with_join;suro_bm25_with_origin_collection;suro_test
        runs_list_suro = []
        run_id_suro='nbbert_ep4_dropout_QWEN_black_bm25_origin_nomessycode_sample3x10-50fromnbrank_top60_dot_500q_batch256_tripledev_4e5'
        for scores, qids, pids in zip(all_logits_suro, all_qids, all_pids):
            sorted_idx = np.array(scores).argsort()[::-1]
            sorted_scores = np.array(scores)[sorted_idx]
            sorted_qids = np.array(qids)[sorted_idx]
            sorted_pids = np.array(pids)[sorted_idx]
            for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
                runs_list_suro.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'BERT-Pair'))
        runs_df_suro = pd.DataFrame(runs_list_suro, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
        runs_df_suro.to_csv(output_dir + '/runs_for_revision/runs.' + run_id_suro + '.' + mode_suro + '.csv', sep='\t', index=False, header=False)


def run_DRmodel_ranking(loss_function = "label-smoothing-cross-entropy", smoothing=0.1, output_dir=grandparent_dir+"/msmarco/train/", mode = 'bm25'):
    # LOAD model
    device = 'cuda'
    from Condenser_model import CondenserForPairwiseModel, CondenserForPairwiseModel_msmarco
    from DenseRetrieval_model import ContrieverForPairwiseModel, ANCEForPairwiseModel,DPRForPairwiseModel
    from LocalEmbedding import localEmbedding_QWEN3
    dr_model_path = model_dir+"/Qwen3-Embedding-4B"
    target_model = localEmbedding_QWEN3(
            dr_model_path,
            device
        )
    target_model.eval()
    target_model.to(device)

    #embedding
    topk = 3
    CON_NAME = model_dir+'/msmarco-bert-co-condensor'
    device = 'cuda'
    # embedding_model = localEmbedding_sentence(
    #         CON_NAME,
    #         device
    # )
    local_data = {}

    print("LOADED ", dr_model_path)

    tokenizer = AutoTokenizer.from_pretrained(dr_model_path)

    all_logits_target = []
    all_labels = []
    all_qids = []
    all_pids = []
    all_examples = []
    cnt = 0
    
    data_class = RAG_Dataset(tokenizer=tokenizer)

    if mode == "bm25":
        # load doc_id=pid to string
        # print("LOADING colletion...")
        # collection_df = pd.read_csv(msmarco_collection_path, sep='\t', names=['docid', 'document_string'])
        # collection_df['docid'] = collection_df['docid'].astype(str)
        # collection_str = collection_df.set_index('docid').to_dict()['document_string']
        # # load query
        # print("LOADING queries...")
        # query_df = pd.read_csv(msmarco_queries_path, names=['qid','query_string'], sep='\t')
        # query_df['qid'] = query_df['qid'].astype(str)
        # queries_str = query_df.set_index('qid').to_dict()['query_string']
        # print(len(queries_str))
        examples, labels, qids, pids = data_class.data_generator_ranking_dr_by_qid(mode=mode)
        all_data = accumulate_list_by_qid_and_pid(examples, pids, qids)
        _ , collection_str, queries_str = msmarco_run_bm25_load() 
    else:
        examples, labels, qids, pids = data_class.data_generator_ranking_dr_by_qid(mode=mode)
        all_data = accumulate_list_by_qid_and_pid(examples, pids, qids)
        queries_str = {}
        collection_str = {}
        for example, qid in zip(examples, qids):
            if qid not in queries_str:
                queries_str[str(qid)] = example[0]
                
        for example, pid in zip(examples, pids):
            if pid not in collection_str:
                collection_str[str(pid)] = example[1]

    num = 0
    for q in tqdm(list(all_data.keys())[:]):
        num+=1
        query = queries_str[str(q)]
        print(num, " - ", query)
        passages = []
        tmp_pid = []
        text_to_did = {}

        
        if mode == "5_up_bm25": 
            for (d, label) in all_data[q]:
                passages.append(collection_str[str(d)])
                tmp_pid.append(d)
                text_to_did[collection_str[str(d)]] = d
        else:
            for d in all_data[q].keys():
                query_ = all_data[q][d][0]
                if query != query_:
                    raise ValueError("Unmatched Query!")
                passages.append(collection_str[str(d)])
                tmp_pid.append(d)
                text_to_did[collection_str[str(d)]] = d


        sim_score , _ = sim_score_for_passage_list(target_model, query, passages, measure="dot", k=len(passages))
        all_examples += [(query, n) for n in passages]
        all_qids += [q]*len(passages)
        all_pids += tmp_pid
        all_logits_target += sim_score

        local_data[q] = []
        # db = FAISS.from_texts(passages, embedding_model, distance_strategy = 'MAX_INNER_PRODUCT')
        # search_result = db.similarity_search(query, k=len(passages), )
        # for t in search_result:
        #     local_data[q].append(text_to_did[t.page_content])
        

        
    # data_new = accumulate_list_by_qid_and_pid(all_examples, all_pids ,all_qids)
    # label_new = accumulate_list_by_qid_and_pid(all_labels, all_pids ,all_qids)
    #accumulates per query
    # all_labels, _ = accumulate_list_by_qid(all_labels, all_qids)
    logit_pid_qid_dict_target = accumulate_list_by_qid_and_pid(all_logits_target, all_pids, all_qids)
    all_logits_target, _ = accumulate_list_by_qid(all_logits_target, all_qids)
    # all_softmax_logits, _ = accumulate_list_by_qid(all_softmax_logits, all_qids)
    all_pids, all_qids = accumulate_list_by_qid(all_pids, all_qids)

    #EXTRA
    # all_logits_local, all_pids_local, all_qids_local, all_labels_local, logit_pid_qid_dict_local = match_and_rank(data_new, label_new ,embedding_model, topk=20)
    
    if mode in ['dev', 'test','bm25']:
        mode = 'target_'+mode+'_dot'
        runs_list = []
        run_id = "DR_QWEN3_4b_bad#_noquery_on_trec_dl_2019" 
        for scores, qids, pids in zip(all_logits_target, all_qids, all_pids):
            sorted_idx = np.array(scores).argsort()[::-1]
            sorted_scores = np.array(scores)[sorted_idx]
            sorted_qids = np.array(qids)[sorted_idx]
            sorted_pids = np.array(pids)[sorted_idx]

            for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
                runs_list.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'BERT-Pair'))
        runs_df = pd.DataFrame(runs_list, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
        runs_df.to_csv(output_dir + '/runs_for_revision/runs.' + run_id + '.' + mode + '.csv', sep='\t', index=False, header=False)

        # mode_suro = 'suro_test'
        # runs_list_suro = []
        # run_id_suro = "epoch80_runbm25_cocondensersample_dot"
        # for scores, qids, pids in zip(all_logits_suro, all_qids, all_pids):
        #     sorted_idx = np.array(scores).argsort()[::-1]
        #     sorted_scores = np.array(scores)[sorted_idx]
        #     sorted_qids = np.array(qids)[sorted_idx]
        #     sorted_pids = np.array(pids)[sorted_idx]
        #     for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
        #         runs_list_suro.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'BERT-Pair'))
        # runs_df_suro = pd.DataFrame(runs_list_suro, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
        # runs_df_suro.to_csv(output_dir + '/runs/runs.' + run_id_suro + '.' + mode_suro + '.csv', sep='\t', index=False, header=False)
        pass

def runs_ranking_similarity(runs_1, runs_2, mode = "test", source_dir = grandparent_dir+"/msmarco/train/runs_for_revision/"):
    print("Reading Runs1 {} ...".format(runs_1))
    data_1 = read_runs(source_dir+runs_1, organize="dict")
    print("Reading Runs2 {} ...".format(runs_2))
    data_2 = read_runs(source_dir+runs_2, organize="dict")

    res = {}
    # rbo = rbo_dict_score(data_1, data_2, p=0.7)
    rbo = avg_rbo(source_dir+runs_1, source_dir+runs_2, topn=1000 , p=0.7)
    # overlap = top_n_overlap_dic_sim(data_1, data_2, topn=10)
    overlap = top_n_overlap(source_dir+runs_1, source_dir+runs_2, topn=10)
    res['inter'] = overlap
    res['rbo'] = rbo

    for metric, v in res.items():
        print("\n{} {} : {:3f}".format(mode, metric, v))

def tsv_to_runs(tsv_path, runs_path, output_dir = "/data_share_from_3090/chenzhuo/msmarco/train/"):
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

    all_logits = []
    all_qids = []
    all_pids = []
    relevant_pairs_dict = collections.defaultdict(list)
    qid_dict = {}
    with open(tsv_path, "r") as f:
        # csv_reader = csv.reader(f)
        for line in tqdm(f):
            
            # line = ''.join(line)
            qid, did, query, passage, label = line.strip().split('\t')
            if qid not in qid_dict:
                qid_dict[qid] = 1
            else:
                qid_dict[qid] += 1
            if did!= 'did' and passage != collection_str[str(did)]:
                raise ValueError("PASSAGE do not match!")

            # qid, did, _ = line.strip().split('\t')
            relevant_pairs_dict[qid].append(did)
            all_logits.append(-qid_dict[qid])
            all_qids.append(qid)
            all_pids.append(did)
    
    all_logits, _ = accumulate_list_by_qid(all_logits, all_qids)
    all_pids, all_qids = accumulate_list_by_qid(all_pids, all_qids)

    mode = 'target_bm25_dot'
    runs_list = []
    run_id = "Run_bm25_allq"
    for scores, qids, pids in zip(all_logits, all_qids, all_pids):
            sorted_idx = np.array(scores).argsort()[::-1]
            sorted_scores = np.array(scores)[sorted_idx]
            sorted_qids = np.array(qids)[sorted_idx]
            sorted_pids = np.array(pids)[sorted_idx]

            for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
                runs_list.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'BERT-Pair'))
    runs_df = pd.DataFrame(runs_list, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
    runs_df.to_csv(output_dir + '/runs/runs.' + run_id + '.' + mode + '.csv', sep='\t', index=False, header=False)

if __name__ == '__main__':
    # test_between_dr_and_surrogate()
    imitation_model_ranking(mode="test")#bm25
    # run_DRmodel_ranking(mode="test")
