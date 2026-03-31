import sys
import os
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import json
import torch
import logging
import time
import bisect
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast, BertConfig, BertModel, AutoModelForSequenceClassification, BertForNextSentencePrediction
from transformers import AutoTokenizer, AutoModel
from opinion_reverse.dense_retrieval.condenser import condenser_encode
# from langchain_community.vectorstores import faiss
import faiss

def opinion_ranking(data,  model, tokenizer, max_seq_len, args, device, device_cpu):
    if args.data_name == "fnc":
        batch_encodings = tokenizer([(e[4], e[3]) for e in data], 
                                    max_length=max_seq_len,padding="max_length", truncation=True, return_tensors='pt')
        labels = [t[2] for t in data]
        true_labels = [t[1] for t in data]
    elif args.data_name == "procon":
        batch_encodings = tokenizer([(e[4], e[3]) for e in data], 
                                    max_length=max_seq_len,padding="max_length", truncation=True, return_tensors='pt')
        labels = [t[1] for t in data]
        true_labels = [t[1] for t in data]
    if args.pat or args.target == 'nb_bert_adv' or args.target == 'mini_im':
            pos_input_ids = batch_encodings['input_ids'].to(device)
            pos_token_type_ids = batch_encodings['token_type_ids'].to(device)
            pos_attention_mask = batch_encodings['attention_mask'].to(device)
            neg_input_ids = batch_encodings['input_ids'].to(device)
            neg_token_type_ids = batch_encodings['token_type_ids'].to(device)
            neg_attention_mask = batch_encodings['attention_mask'].to(device)
            outputs = model(
                    input_ids_pos=pos_input_ids,
                    attention_mask_pos=pos_attention_mask,
                    token_type_ids_pos=pos_token_type_ids,
                    input_ids_neg=neg_input_ids,
                    attention_mask_neg=neg_attention_mask,
                    token_type_ids_neg=neg_token_type_ids,
                )
            
            outputs[0].to(device_cpu)
            all_logits = outputs[0][:,1].tolist()
            # all_logits.append(logits.cpu().detach().numpy())
    elif args.target == "nb_bert":
            input_ids = batch_encodings['input_ids'].to(device)
            token_type_ids = batch_encodings['token_type_ids'].to(device)
            attention_mask = batch_encodings['attention_mask'].to(device)
            outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            outputs[0].to(device_cpu)
            all_logits = outputs[0][:,1].tolist()
    elif args.target == "mini":
        input_ids = batch_encodings['input_ids'].to(device)
        token_type_ids = batch_encodings['token_type_ids'].to(device)
        attention_mask = batch_encodings['attention_mask'].to(device)
        outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
        outputs[0].to(device_cpu)
        all_logits = outputs[0][:,0].tolist()
    sorted_index = np.argsort(-np.array(all_logits))
    sorted_label = [labels[t] for t in list(sorted_index)]
    sorted_logits = [all_logits[t] for t in list(sorted_index)]
    sorted_true_label = [true_labels[t] for t in list(sorted_index)]
    sorted_data = [data[t] for t in list(sorted_index)]
    print(sorted_index, ":sorted_vlabels:", sorted_label)
    print("Logit:", sorted_logits)
    
    return sorted_index, sorted_label, sorted_data, sorted_true_label, sorted_logits

def batching(tokenizer, examples, args):
     for i in tqdm(range(0, len(examples), args.batch_size), desc='Processing:'):
            tmp_examples = examples[i: i+args.batch_size]
            tmp_labels = [t[1] for t in tmp_examples]
            tmp_true_labels = [t[1] for t in tmp_examples]

            batch_encoding_queries = tokenizer([e[4] for e in tmp_examples], max_length=args.max_seq_len,padding="max_length", truncation=True, return_tensors='pt')
            # batch_encoding_queries = tokenizer([e[4] for e in tmp_examples])
            # batch_encoding_queries = torch.tensor(batch_encoding_queries)
            batch_encoding_passages = tokenizer([e[3] for e in tmp_examples], max_length=args.max_seq_len, padding = 'max_length', truncation=True, return_tensors='pt')
            # batch_encoding_passages = tokenizer([e[3] for e in tmp_examples])['input_ids']
            # batch_encoding_passages = torch.tensor(batch_encoding_passages)
            # batch_encoding_neg = self.tokenizer([(e[0], e[2]) for e in tmp_examples], 
            #                         max_length=max_seq_len,padding="max_length", truncation=True, return_tensors='pt')
            yield batch_encoding_queries, batch_encoding_passages, tmp_labels, tmp_true_labels

def dense_dual_encoder_ranking(data,  model, tokenizer, max_seq_len, args, device, target_info=None):
    if args.data_name == "fnc":
        batch_encodings = [(e[4], e[3]) for e in data] 
        labels = [t[2] for t in data]
        true_labels = [t[1] for t in data]
    elif args.data_name == "procon" and args.target != 'bge' and args.batch_split:
        labels = [t[1] for t in data]
        true_labels = [t[1] for t in data]
    elif args.data_name == "procon" and args.target != 'bge':
        queries_encodings = tokenizer([e[4] for e in data], max_length=max_seq_len,padding="max_length", truncation=True, return_tensors='pt') 
        passage_encodings = tokenizer([e[3] for e in data], max_length=max_seq_len,padding="max_length", truncation=True, return_tensors='pt')
        labels = [t[1] for t in data]
        true_labels = [t[1] for t in data]
    elif args.data_name == "procon":
        queries = [e[4] for e in data]
        passages = [e[3] for e in data]
        labels = [t[1] for t in data]
        true_labels = [t[1] for t in data]

    if args.pat and (args.target == 'condenser') and args.batch_split:
        all_logits = []
        data_object = batching(tokenizer=tokenizer, examples=data, args=args)
        for batch_encoding_queries, batch_encoding_passages, tmp_labels, tmp_true_labels in data_object:
            outputs = model(
             query = batch_encoding_queries,
             pos = batch_encoding_passages,
             neg = batch_encoding_passages,
            )
            """outputs = model(
             input_ids_query = batch_encoding_queries['input_ids'].to(device),
             input_ids_pos = batch_encoding_passages['input_ids'].to(device),
             input_ids_neg = batch_encoding_passages['input_ids'].to(device),
            )"""
            logits = outputs[0].tolist()
            all_logits.extend(logits)
             
    elif args.pat and args.target == 'condenser':
        outputs = model(
             input_ids_query = queries_encodings['input_ids'].to(device),
             input_ids_pos = passage_encodings['input_ids'].to(device),
             input_ids_neg = passage_encodings['input_ids'].to(device),
        )
        all_logits = outputs[0].tolist()
    elif args.target == "bge":
            outputs = model(
                    queries=queries,
                    passages=passages,
                )
            # outputs[0].to(device_cpu)
            all_logits = outputs[0].tolist()
    elif args.target == 'condenser':
        all_logits = []
        data_object = batching(tokenizer=tokenizer, examples=data, args=args)
        for batch_encoding_queries, batch_encoding_passages, tmp_labels, tmp_true_labels in data_object:
            # outputs = condenser_encode(
            #     model,
            #     batch_encoding_queries,
            #     batch_encoding_passages,
            #     device,
            #     args,
            # )
            outputs, _ = dense_L2_retrieval(model, batch_encoding_queries, batch_encoding_passages, k=len(tmp_labels))
            logits = outputs[0].tolist()
            all_logits.extend(logits)

    if target_info is not None:
        BOOST_RANK = 0
        print("##### COMPARISON ######")
        for t in target_info:
             ori_rank = t[0]
             old_scores = t[1]
             old_score = old_scores[ori_rank]
             reverse = sorted(old_scores)
             new_rank = len(old_scores) - bisect.bisect_left(reverse, all_logits[ori_rank])
             BOOST_RANK += (ori_rank - new_rank)
             print(old_score ,"vs", all_logits[ori_rank], "ori_rank:", ori_rank, "new_rank:", new_rank)
        print("TOPIC BR:", BOOST_RANK/len(target_info))

    sorted_index = np.argsort(-np.array(all_logits))
    sorted_label = [labels[t] for t in list(sorted_index)]
    sorted_logits = [all_logits[t] for t in list(sorted_index)]
    sorted_true_label = [true_labels[t] for t in list(sorted_index)]
    sorted_data = [data[t] for t in list(sorted_index)]
    print(sorted_index, "sorted_logits:", sorted_logits,":sorted_vlabels:", sorted_label)
    
    return sorted_index, sorted_label, sorted_data, sorted_true_label, sorted_logits

def eval_ranking(data,  model, tokenizer, max_seq_len, args, device, target_info=None):
    if args.data_name == "fnc":
        batch_encodings = [(e[4], e[3]) for e in data] 
        labels = [t[2] for t in data]
        true_labels = [t[1] for t in data]
    elif args.data_name == "procon" and args.target != 'bge' and args.batch_split and args.eval_model != "ance" and args.eval_model != "QWEN":
        labels = [t[1] for t in data]
        true_labels = [t[1] for t in data]
    elif args.data_name == "procon" and args.target != 'bge' and args.eval_model != "ance" and args.eval_model != "QWEN":
        queries_encodings = tokenizer([e[4] for e in data], max_length=max_seq_len,padding="max_length", truncation=True, return_tensors='pt')
        passage_encodings = tokenizer([e[3] for e in data], max_length=max_seq_len,padding="max_length", truncation=True, return_tensors='pt')
        labels = [t[1] for t in data]
        true_labels = [t[1] for t in data]
    elif args.data_name == "procon":
        queries = [e[4] for e in data]
        passages = [e[3] for e in data]
        labels = [t[1] for t in data]
        true_labels = [t[1] for t in data]

    if (args.pat or args.query_plus) and (args.target == 'condenser' or args.eval_on_other) and args.batch_split:
        all_logits = []
        if args.eval_model != "ance" and args.eval_model != "QWEN":
            data_object = batching(tokenizer=tokenizer, examples=data, args=args)
            for batch_encoding_queries, batch_encoding_passages, tmp_labels, tmp_true_labels in data_object:
                # outputs = model(
                #  query = batch_encoding_queries,
                #  pos = batch_encoding_passages,
                #  neg = batch_encoding_passages,
                # )
                """outputs = model(
                input_ids_query = batch_encoding_queries['input_ids'].to(device),
                input_ids_pos = batch_encoding_passages['input_ids'].to(device),
                input_ids_neg = batch_encoding_passages['input_ids'].to(device),
                )"""
                outputs, _ = dense_L2_retrieval(model, batch_encoding_queries, batch_encoding_passages, k=len(tmp_labels))
                logits = outputs[0].tolist()
                all_logits.extend(logits)
        else:
            outputs, _ = dense_L2_retrieval(model, queries, passages, k=len(labels))
            all_logits = outputs[0].tolist()
             
    elif args.pat and args.target == 'condenser':
        outputs = model(
             input_ids_query = queries_encodings['input_ids'].to(device),
             input_ids_pos = passage_encodings['input_ids'].to(device),
             input_ids_neg = passage_encodings['input_ids'].to(device),
        )
        all_logits = outputs[0].tolist()
    elif args.target == "bge":
            outputs = model(
                    queries=queries,
                    passages=passages,
                )
            # outputs[0].to(device_cpu)
            all_logits = outputs[0].tolist()
    elif args.target == 'condenser':
        all_logits = []
        data_object = batching(tokenizer=tokenizer, examples=data, args=args)
        for batch_encoding_queries, batch_encoding_passages, tmp_labels, tmp_true_labels in data_object:
            outputs = condenser_encode(
                model,
                batch_encoding_queries,
                batch_encoding_passages,
                device,
                args,
            )
            logits = outputs[0].tolist()
            all_logits.extend(logits)

    sorted_index = np.argsort(-np.array(all_logits))
    sorted_label = [labels[t] for t in list(sorted_index)]
    sorted_logits = [all_logits[t] for t in list(sorted_index)]
    sorted_true_label = [true_labels[t] for t in list(sorted_index)]
    sorted_data = [data[t] for t in list(sorted_index)]
    print("#OTHER EVAL:")
    print("sorted_logits:", sorted_logits)
    print("sorted_vlabels:", sorted_label)
    print("###")
    
    return sorted_index, sorted_label, sorted_data, sorted_true_label, sorted_logits

def dense_L2_retrieval(model, query, candidates, k=10):
    """
    INPUT:
    dense model;str;list
    """

    query_emb = model.encode_(query)[0]
    doc_embs = model.encode_(candidates)
    if type(query_emb) != np.ndarray:
         query_emb = query_emb.cpu().numpy()
         doc_embs = doc_embs.cpu().numpy()
    d = len(query_emb)
    query_emb = np.array([query_emb])

    # index = faiss.IndexFlatL2(d)
    index = faiss.IndexFlatIP(d)
    index.add(doc_embs) 
    dis, ind = index.search(query_emb, k)
    dis_score = [0]*len(dis[0])
    for i in range(len(dis_score)):
         dis_score[ind[0][i]] = dis[0][i]
    
    return np.array([dis_score]), ind

