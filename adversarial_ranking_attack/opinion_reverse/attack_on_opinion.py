import sys
import os
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)
import json
import torch
import logging
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast, BertConfig, BertModel, AutoModelForSequenceClassification, BertForNextSentencePrediction
from transformers import AutoTokenizer, AutoModel
from bert_ranker.models import pairwise_miniLM, pairwise_NB_bert, pairwise_bert, pairwise_NB_bert_classifier
from bert_ranker.models.modeling import RankingBERT_Train, RankingBERT_Pairwise, RankBertForPairwise
from adv_ir.collision_point import gen_aggressive_collision,gen_natural_collision
from adv_ir.attack_methods import pairwise_anchor_trigger, pairwise_anchor_trigger_for_im
from adv_ir.filtering_trigger import eval_rank_change_without_utils
from bert_ranker.models.pairwise_NB_bert import NBBERTForPairwiseLearning
from opinion_reverse.dense_retrieval.ranking_attacks import gen_natural_collision_DR, pairwise_anchor_trigger_DR
import torch
import torch.nn as nn
import collections
import argparse
import random
from ir.scorer import SentenceScorer
# from apex import amp
from nltk.corpus import stopwords

def attack_opinion_with_triggers(model, lm_model, eval_lm_model, nsp_model, tokenizer, query ,target_p_list, anchor_p_list, device, args):
    """
    lm_model = BertForLM.from_pretrained(args.lm_model_dir)
    lm_model.to(device)
    lm_model.eval()
    for param in lm_model.parameters():
        param.requires_grad = False"""
    if args.pat:
        for param in model.parameters():
            param.requires_grad = False
        # model, lm_model, nsp_model = amp.initialize([model, lm_model, nsp_model])
        # eval_lm_model = SentenceScorer(device)
        vocab = tokenizer.vocab
        words = [w for w in vocab if w.isalpha() and w not in set(stopwords.words('english'))]
    else:
        for param in model.parameters():
            param.requires_grad = False
        # if args.target == "bge":
        #     # lm_model= amp.initialize(lm_model)
        #     lm_model.encoder.model= amp.initialize(lm_model.encoder.model)
        #     # pass
        # else:
        #     model, lm_model= amp.initialize([model, lm_model])
        # eval_lm_model = SentenceScorer(device)
    triggers = []
    if args.pat:
         for j in tqdm(range(len(target_p_list))):#target_p_list:[[passage, query], id, origin_id]
             trigger, new_score, trigger_cands = pairwise_anchor_trigger_for_im(
                        query=query,
                        anchor=anchor_p_list[j],
                        raw_passage=target_p_list[j][0][0],
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        words=words,
                        args=args,
                        lm_model=lm_model,
                        nsp_model=nsp_model)
             triggers.append(trigger)
             target_p_list[j][0][0] = trigger+" "+target_p_list[j][0][0]
    elif args.nature:
        trigger, new_score, trigger_cands = gen_natural_collision(
                            inputs_a=query,
                            inputs_b="",#best_sent,
                            model=model, 
                            tokenizer=tokenizer, 
                            device=device, 
                            lm_model=lm_model,  
                            eval_lm_model=eval_lm_model, 
                            args=args
                        )
        triggers = [trigger]*len(target_p_list)
        target_p_list = [[[trigger+" "+t[0][0],t[0][1]] , t[1]] for t in target_p_list]
    elif args.query_plus:
        triggers = [query]*len(target_p_list)
        target_p_list = [[[query+" "+t[0][0],t[0][1]] , t[1]] for t in target_p_list]#target_p_list:[[triggered passage, query], id]
    else:
        trigger, new_score, trigger_cands = gen_aggressive_collision(
                inputs_a=query, 
                inputs_b="", 
                model=model, 
                tokenizer=tokenizer,
                device=device, 
                margin=None, 
                lm_model=lm_model, 
                args=args
            )
        triggers = [trigger]*len(target_p_list)
        target_p_list = [[[trigger+" "+t[0][0],t[0][1]] , t[1]] for t in target_p_list]#article；index
    print("trigger:" ,triggers)
    queries = [query]*(len(target_p_list))
    torch.cuda.empty_cache() 
    return target_p_list, triggers