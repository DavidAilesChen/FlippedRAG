import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
import torch
from torch import cuda
import json
import bisect
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
# from apex import amp
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    BertForNextSentencePrediction,
)
from adv_ir.attack_methods import pairwise_anchor_trigger

device = 'cuda' if cuda.is_available() else 'cpu'

def test_trigger_success_or_not_by_tails(qid, query, trigger, pos_neg_list, query_scores, target_q_passage, passages_dict, collection_str, tokenizer, model, args):
    success_triples = []
    old_scores = query_scores[qid][::-1]

    boost_rank = 0
    for did in target_q_passage[qid]:
        tmp_best_new_score = -1e9
        old_rank, raw_score, label = target_q_passage[qid][did]
                
        # head
        triggered_passage = trigger + ' ' + passages_dict[did]
        # triggered_passage = query + ' ' + passages_dict[did]
        # tail
        # triggered_passage = passages_dict[did] + ' ' + trigger
        #half_len_passage = int(len(passages_dict[did]) / 2)
        #triggered_passage = passages_dict[did][:half_len_passage] + ' ' + trigger + ' ' + passages_dict[did][half_len_passage:]
                              
        batch_encoding = tokenizer([[query, triggered_passage]], max_length=256, padding="max_length", truncation=True, return_tensors='pt')
                
        if args.target == 'mini':
            outputs = model(**(batch_encoding.to(device)))
            new_score = outputs.logits.squeeze().item()
        elif args.target == 'nb_bert':
            outputs = model(**(batch_encoding.to(device)))
            print(outputs.logits.squeeze())
            new_score = outputs.logits.squeeze()[1].item()
        elif args.target == 'mini_adv':
            pos_input_ids = batch_encoding['input_ids'].to(device)
            pos_token_type_ids = batch_encoding['token_type_ids'].to(device)
            pos_attention_mask = batch_encoding['attention_mask'].to(device)
            neg_input_ids = batch_encoding['input_ids'].to(device)
            neg_token_type_ids = batch_encoding['token_type_ids'].to(device)
            neg_attention_mask = batch_encoding['attention_mask'].to(device)
            outputs = model(
                input_ids_pos=pos_input_ids,
                attention_mask_pos=pos_attention_mask,
                token_type_ids_pos=pos_token_type_ids,
                input_ids_neg=neg_input_ids,
                attention_mask_neg=neg_attention_mask,
                token_type_ids_neg=neg_token_type_ids,
                )
            new_score = outputs[0][0, -1].item()
        elif args.target == 'large':
            outputs = model(**(batch_encoding.to(device)))
            new_score = outputs.logits[0, -1].item()
        else:
            input_ids = batch_encoding['input_ids'].to(device)
            token_type_ids = batch_encoding['token_type_ids'].to(device)
            attention_mask = batch_encoding['attention_mask'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=token_type_ids,
                            token_type_ids=attention_mask)[0]
            new_score = outputs[0, 1].item()

        print("Trigger: {}".format(trigger))

        new_rank = len(old_scores) - bisect.bisect_left(old_scores, new_score)
        #rank_list.append(new_rank)
        boost_rank+=(old_rank-new_rank)
            
        print(f'Query id={qid}, Doc id={did}, '
                    f'old score={raw_score:.4f}, new score={new_score:.4f}, old rank={old_rank}, new rank={new_rank}')
    if boost_rank>0:
        for j in pos_neg_list:
            pos_text = collection_str[j[0]]
            neg_text = collection_str[j[2]]
            success_triples.append([query, pos_text, trigger+' '+neg_text])
    return success_triples

def eval_rank_change_passage_wise(qid, did, query, passage, trigger, query_scores, target_q_passage, passages_dict, tokenizer, model, args):
    old_scores = query_scores[qid][::-1]

    boost_rank = 0
    tmp_best_new_score = -1e9
    old_rank, raw_score, label = target_q_passage[qid][did]
    if trigger is None:
        adv_sample = passage
    else:
        adv_sample = trigger + ' ' + passages_dict[did]
    batch_encoding = tokenizer([(query, adv_sample)], max_length=128, padding="max_length", truncation=True, return_tensors='pt')
    ori_encoding = tokenizer([(query, passages_dict[did])], max_length=128, padding="max_length", truncation=True, return_tensors='pt')

    if args.target == 'mini' :
        outputs = model(**(batch_encoding.to(device)))
        new_score = outputs.logits.squeeze().item()
    elif args.target == "bert_rank":
        input_ids = batch_encoding['input_ids'].to(device)
        token_type_ids = batch_encoding['token_type_ids'].to(device)
        ori_input_ids = ori_encoding['input_ids'].to(device)
        ori_token_type_ids = ori_encoding['token_type_ids'].to(device)
        if args.pairwise:
            outputs = model(
                    input_ids_pos=input_ids,
                    token_type_ids_pos=token_type_ids,
                    input_ids_neg=input_ids,
                    token_type_ids_neg=token_type_ids
                )
            ori_outputs = model(
                    input_ids_pos=ori_input_ids,
                    token_type_ids_pos=ori_token_type_ids,
                    input_ids_neg=ori_input_ids,
                    token_type_ids_neg=ori_token_type_ids
                )
        else:
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids
            )
            ori_outputs = model(
                input_ids=ori_input_ids,
                token_type_ids=ori_token_type_ids
            )
        new_score = outputs[0, 0].item()
        ori_score = ori_outputs[0 ,0].item()
    elif args.target == 'nb_bert':
        outputs = model(**(batch_encoding.to(device)))
        print(outputs.logits.squeeze())
        new_score = outputs.logits.squeeze()[1].item()
    elif args.target == 'mini_adv' or args.target == 'nb_bert_pair':
        pos_input_ids = batch_encoding['input_ids'].to(device)
        pos_token_type_ids = batch_encoding['token_type_ids'].to(device)
        pos_attention_mask = batch_encoding['attention_mask'].to(device)
        neg_input_ids = batch_encoding['input_ids'].to(device)
        neg_token_type_ids = batch_encoding['token_type_ids'].to(device)
        neg_attention_mask = batch_encoding['attention_mask'].to(device)
        outputs = model(
            input_ids_pos=pos_input_ids,
            attention_mask_pos=pos_attention_mask,
            token_type_ids_pos=pos_token_type_ids,
            input_ids_neg=neg_input_ids,
            attention_mask_neg=neg_attention_mask,
            token_type_ids_neg=neg_token_type_ids,
            )
        new_score = outputs[0][0, -1].item()
    elif args.target == 'large':
        outputs = model(**(batch_encoding.to(device)))
        new_score = outputs.logits[0, -1].item()
    else:
        input_ids = batch_encoding['input_ids'].to(device)
        token_type_ids = batch_encoding['token_type_ids'].to(device)
        attention_mask = batch_encoding['attention_mask'].to(device)

        outputs = model(input_ids=input_ids,
                            attention_mask=token_type_ids,
                            token_type_ids=attention_mask)[0]
        new_score = outputs[0, 1].item()

    new_rank = len(old_scores) - bisect.bisect_left(old_scores, new_score)
    #rank_list.append(new_rank)
    boost_rank+=(old_rank-new_rank)
    success = 0
    if boost_rank>0:
        success=1
            
    print(f'Query id={qid}, Doc id={did}, '
                    f'old score={raw_score:.4f}, new score={new_score:.4f}, old rank={old_rank}, new rank={new_rank}, ori_score={ori_score}')
    return boost_rank, success, new_rank
    


def test_trigger_success_or_not_between_triple(qid, query, trigger, pos_neg_list, collection_str, query_scores,tokenizer, model, args):
    success_triples = []
    old_scores = query_scores[qid][::-1]
    for i in range(len(pos_neg_list)):
        pos_id = pos_neg_list[i][0]
        neg_id = pos_neg_list[i][2]
        pos_rank = pos_neg_list[i][1]
        neg_rank = pos_neg_list[i][3]
        if pos_rank>=neg_rank:
            raise ValueError("POS is worse than NEG!")
        pos = collection_str[pos_id]
        neg = collection_str[neg_id]
        neg_adv = trigger+" "+neg

        batch_encoding = tokenizer([[query, neg_adv]], max_length=256, padding="max_length", truncation=True, return_tensors='pt')
        if args.target == 'mini':
            outputs = model(**(batch_encoding.to(device)))
            new_score = outputs.logits.squeeze().item()
        elif args.target == 'nb_bert':
            outputs = model(**(batch_encoding.to(device)))
            print("nbbert_out:",outputs.logits.squeeze())
            new_score = outputs.logits.squeeze()[1].item()
        elif args.target == 'mini_adv':
            pos_input_ids = batch_encoding['input_ids'].to(device)
            pos_token_type_ids = batch_encoding['token_type_ids'].to(device)
            pos_attention_mask = batch_encoding['attention_mask'].to(device)
            neg_input_ids = batch_encoding['input_ids'].to(device)
            neg_token_type_ids = batch_encoding['token_type_ids'].to(device)
            neg_attention_mask = batch_encoding['attention_mask'].to(device)
            outputs = model(
                input_ids_pos=pos_input_ids,
                attention_mask_pos=pos_attention_mask,
                token_type_ids_pos=pos_token_type_ids,
                input_ids_neg=neg_input_ids,
                attention_mask_neg=neg_attention_mask,
                token_type_ids_neg=neg_token_type_ids,
                )
            new_score = outputs[0][0, -1].item()
        elif args.target == 'large':
            outputs = model(**(batch_encoding.to(device)))
            new_score = outputs.logits[0, -1].item()
        else:
            input_ids = batch_encoding['input_ids'].to(device)
            token_type_ids = batch_encoding['token_type_ids'].to(device)
            attention_mask = batch_encoding['attention_mask'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=token_type_ids,
                            token_type_ids=attention_mask)
            out = outputs[0]
            print(outputs)
            new_score = out[0, 1].item()
            print("newscore:", new_score)
            sys.exit(0)
        
        print("Trigger: {}".format(trigger))
        
        new_rank = len(old_scores) - bisect.bisect_left(old_scores, new_score)
        print(f'Query id={qid}, Doc id={neg_id}, '
                    f'new score={new_score:.4f}, old rank={neg_rank}, new rank={new_rank}')

        if new_rank<pos_rank:
            success_triples.append([query, pos, neg_adv])
    
    return success_triples

def test_pat_trigger_success_or_not_between_triple(qid, query,  pos_neg_list, words, collection_str, query_scores,tokenizer, model, lm_model, nsp_model,device, args):
    success_triples = []
    success_triggers = []
    old_scores = query_scores[qid][::-1]
    for i in range(len(pos_neg_list)):
        pos_id = pos_neg_list[i][0]
        neg_id = pos_neg_list[i][2]
        pos_rank = pos_neg_list[i][1]
        neg_rank = pos_neg_list[i][3]
        if pos_rank>=neg_rank:
            raise ValueError("POS is worse than NEG!")
        anchor = collection_str[pos_id]
        raw_passage = collection_str[neg_id]

        trigger, new_score, trigger_cands = pairwise_anchor_trigger(
                    query=query,
                    anchor=anchor,
                    raw_passage=raw_passage,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    words=words,
                    args=args,
                    lm_model=lm_model,
                    nsp_model=nsp_model)

        neg_adv = trigger+" "+raw_passage

        batch_encoding = tokenizer([[query, neg_adv]], max_length=256, padding="max_length", truncation=True, return_tensors='pt')
        if args.target == 'mini':
            outputs = model(**(batch_encoding.to(device)))
            new_score = outputs.logits.squeeze().item()
        elif args.target == 'nb_bert':
            pos_input_ids = batch_encoding['input_ids'].to(device)
            pos_token_type_ids = batch_encoding['token_type_ids'].to(device)
            pos_attention_mask = batch_encoding['attention_mask'].to(device)
            neg_input_ids = batch_encoding['input_ids'].to(device)
            neg_token_type_ids = batch_encoding['token_type_ids'].to(device)
            neg_attention_mask = batch_encoding['attention_mask'].to(device)
            outputs = model(
                input_ids_pos=pos_input_ids,
                attention_mask_pos=pos_attention_mask,
                token_type_ids_pos=pos_token_type_ids,
                input_ids_neg=neg_input_ids,
                attention_mask_neg=neg_attention_mask,
                token_type_ids_neg=neg_token_type_ids,
            )
            new_score = outputs[0][0, -1].item()
            print("nbbert_out:",outputs[0].squeeze())
        elif args.target == 'large':
            outputs = model(**(batch_encoding.to(device)))
            new_score = outputs.logits[0, -1].item()
        else:
            input_ids = batch_encoding['input_ids'].to(device)
            token_type_ids = batch_encoding['token_type_ids'].to(device)
            attention_mask = batch_encoding['attention_mask'].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=token_type_ids,
                            token_type_ids=attention_mask)
            out = outputs[0]
            print(outputs)
            new_score = out[0, 1].item()
            print("newscore:", new_score)
            sys.exit(0)
        
        print("Trigger: {}".format(trigger))

        new_rank = len(old_scores) - bisect.bisect_left(old_scores, new_score)
        print(f'Query id={qid}, Doc id={neg_id}, '
                    f'new score={new_score:.4f}, old rank={neg_rank}, new rank={new_rank}')
        if new_rank>60:
            break

        if new_rank<pos_rank:
            success_triples.append([query, anchor, neg_adv])
            success_triggers.append(trigger)
    
    return success_triples, success_triggers

def eval_rank_change_without_utils(model, tokenizer, queries, triggers, target_passages, ori_scores, device, args):
    success = 0
    boost_rank = 0
    for i in range(len(target_passages)):
        old_rank = list(target_passages[i].values())[0][0]
        raw_score = list(target_passages[i].values())[0][1]
        adv_sample = triggers[i]+ ' '+ list(target_passages[i].keys())[0]
        encoding = tokenizer([(queries[i], adv_sample)],max_length=args.max_seq_len, padding="max_length", truncation=True, return_tensors='pt')
        if args.target == 'nb_bert_adv':
            pos_input_ids = encoding['input_ids'].to(device)
            pos_token_type_ids = encoding['token_type_ids'].to(device)
            pos_attention_mask = encoding['attention_mask'].to(device)
            neg_input_ids = encoding['input_ids'].to(device)
            neg_token_type_ids = encoding['token_type_ids'].to(device)
            neg_attention_mask = encoding['attention_mask'].to(device)
            outputs = model(
                    input_ids_pos=pos_input_ids,
                    attention_mask_pos=pos_attention_mask,
                    token_type_ids_pos=pos_token_type_ids,
                    input_ids_neg=neg_input_ids,
                    attention_mask_neg=neg_attention_mask,
                    token_type_ids_neg=neg_token_type_ids,
                )
            logits = outputs[0][0,1].item()
        elif args.target == 'nb_bert':
            if args.pat:
                pos_input_ids = encoding['input_ids'].to(device)
                pos_token_type_ids = encoding['token_type_ids'].to(device)
                pos_attention_mask = encoding['attention_mask'].to(device)
                neg_input_ids = encoding['input_ids'].to(device)
                neg_token_type_ids = encoding['token_type_ids'].to(device)
                neg_attention_mask = encoding['attention_mask'].to(device)
                outputs = model(
                    input_ids_pos=pos_input_ids,
                    attention_mask_pos=pos_attention_mask,
                    token_type_ids_pos=pos_token_type_ids,
                    input_ids_neg=neg_input_ids,
                    attention_mask_neg=neg_attention_mask,
                    token_type_ids_neg=neg_token_type_ids,
                )
                logits = outputs[0][0,1].item()
            else:
                input_ids = encoding['input_ids'].to(device)
                token_type_ids = encoding['token_type_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs[0][0,1].item()
        new_rank = len(ori_scores) - bisect.bisect_left(ori_scores, logits)
        #rank_list.append(new_rank)
        boost_rank+=(old_rank-new_rank)
        if boost_rank>0:
            success += 1
            
        print(f'old score={raw_score:.4f}, new score={logits:.4f}, old rank={old_rank}, new rank={new_rank}')

    return success, boost_rank, 

        
