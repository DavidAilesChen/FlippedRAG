"""
Single Query Attack method target ponitwise ranker on MSMARCO Passage Ranking
Using the generate triggers to test its transferability
"""
import sys
import os

#from sqlalchemy import true
from torch import true_divide

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)
prodir = '/data_2/chenzhuo/adversarial_ranking_attack/bert_ranker/results'
prodir2 = "/data_share/chenzhuo/adversarial_data/results"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import torch
import logging
import time

import torch.optim as optim
import numpy as np
import random
import argparse
import bisect
from tqdm import tqdm
from torch.autograd import Variable
from torch import cuda
import torch.nn.functional as F
from copy import deepcopy

# from pattern.text.en import singularize, pluralize
from transformers import BertTokenizer, BertTokenizerFast, BertForNextSentencePrediction, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel
from bert_ranker.models import pointwise_bert, pairwise_bert, pairwise_miniLM, pairwise_NB_bert
from bert_ranker.metrics import evaluate_and_aggregate, metrics
from bert_ranker import bert_ranker_utils
from ir.bert_models import BertForLM
from ir.scorer import SentenceScorer
from apex import amp
from data_utils import prepare_data_and_scores
from collision_point import gen_aggressive_collision, gen_natural_collision

device = 'cuda:0' if cuda.is_available() else 'cpu'
K = 10
BIRCH_DIR = prodir + '/data/birch'
BIRCH_MODEL_DIR = BIRCH_DIR + '/models'
BIRCH_DATA_DIR = BIRCH_DIR + '/data'
BIRCH_ALPHAS = [1.0, 0.5, 0.1]
BIRCH_GAMMA = 0.6
BERT_LM_MODEL_DIR = '/data_1/ljw/pat/data/wiki103/bert/'
BOS_TOKEN = '[unused0]'

def main():
    parser = argparse.ArgumentParser('Collision_Attack')

    parser.add_argument('--mode', default='test', type=str,
                        help='train/test')

    # target known model config
    parser.add_argument("--experiment_name", default='collision.pointwise', type=str)
    parser.add_argument("--target", type=str, default='mini', help='test on what model')
    parser.add_argument("--target_type", type=str, default='none', help='target model of what kind of trigger')

    parser.add_argument("--data_name", default="dl", type=str)
    parser.add_argument("--method", default="nature", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--model_path", default="/data_share/chenzhuo/adversarial_data/model/nbbert_embedding_adv.pt", type=str)
    parser.add_argument("--transformer_model", default="cross-encoder/ms-marco-MiniLM-L-12-v2", type=str, required=False, help="Bert model to use (cross-encoder/ms-marco-MiniLM-L-12-v2,bert-base-uncased).")
    parser.add_argument('--stemp', type=float, default=1.0, help='temperature of softmax')
    parser.add_argument('--lr', type=float, default=0.001, help='optimization step size')
    parser.add_argument('--max_iter', type=int, default=5, help='maximum iteraiton')
    parser.add_argument('--seq_len', type=int, default=6, help='Sequence length')
    parser.add_argument('--min_len', type=int, default=5, help='Min sequence length')
    parser.add_argument("--beta", default=0., type=float, help="Coefficient for language model loss.")
    parser.add_argument("--amount", default=0, type=int, help="adv_Data amount.")
    parser.add_argument('--save', action='store_true', help='Save collision to file')
    parser.add_argument('--verbose', action='store_true', default=True,  help='Print every iteration')
    parser.add_argument("--lm_model_dir", default=BERT_LM_MODEL_DIR, type=str, help="Path to pre-trained language model")
    parser.add_argument('--perturb_iter', type=int, default=5, help='PPLM iteration')
    parser.add_argument("--kl_scale", default=0.0, type=float, help="KL divergence coefficient")
    parser.add_argument("--topk", default=50, type=int, help="Top k sampling for beam search")
    parser.add_argument("--num_beams", default=5, type=int, help="Number of beams")
    parser.add_argument("--num_filters", default=100, type=int, help="Number of num_filters words to be filtered")
    parser.add_argument('--nature', action='store_true', help='Nature collision')
    parser.add_argument('--pat', action='store_true', help='PAT.')
    parser.add_argument('--regularize', action='store_true', help='Use regularize to decrease perplexity')
    parser.add_argument('--fp16', default=True, action='store_true', help='fp16')
    parser.add_argument('--patience_limit', type=int, default=2, help="Patience for early stopping.")
    parser.add_argument("--seed", default=42, type=str, help="random seed")
    
    # nature: True True beta=0.015 stemp=0.02, num_beams=10, topk=150, max_iter=5
    # python run_collision.py --nature --beta=0.02 --stemp=0.1 --num_beams=1 --topk=50 --max_iter=5 --mode=train
    # constrains: True; False beta=0.85 stemp=1.0, num_beams=5, topk=40, max_iter=30
    # python run_collision.py --regularize --beta=0.85 --stemp=1.0 --num_beams=5 --topk=40  --max_iter=30 --mode=train
    # aggressive: False; False  beta=0 stemp=1.0, num_beams=5, topk=50, max_iter=30
    # python run_collision.py --beta=0.0 --stemp=1.0 --num_beams=5 --topk=50 --max_iter=30 --mode=train

    args = parser.parse_args()

    #tokenizer = BertTokenizerFast.from_pretrained(args.transformer_model)
    if args.transformer_model == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    else:
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    if args.mode == 'train':
        gen_collision(args, tokenizer)
    elif args.mode == 'test':
        test_conllision(args)
    elif args.mode == 'test_ranking':
        test_conllision_ranking(args, tokenizer)
    else:
        raise ValueError('Not implemented error!')


def gen_collision(args, tokenizer):
    logger = logging.getLogger("Pytorch")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    #log_dir = curdir + '/logs'
    log_dir = prodir2 + '/collison_logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    now_time = '_'.join(time.asctime(time.localtime(time.time())).split()[:3])
    log_path = log_dir + '/collision_on_{}.{}.{}.{}.{}.{}.{}.log'.format(args.target, args.regularize, args.beta, args.nature, args.method , args.target_type , now_time)

    if os.path.exists(log_path):
        os.remove(log_path)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Runing with configurations: {}'.format(json.dumps(args.__dict__, indent=4)))
    print('Runing with configurations: {}'.format(json.dumps(args.__dict__, indent=4)))

    target_q_passage, query_scores, best_query_sent, queries, passages_dict = prepare_data_and_scores(target_name=args.target,
                                                                                    data_name=args.data_name,
                                                                                    top_k=5,
                                                                                    least_num=5,
                                                                                    target_type=args.target_type)
    """
    model = pointwise_bert.BertForPointwiseLearning.from_pretrained(args.transformer_model)
    model.to(device)
    model_path = prodir + '/bert_ranker/saved_models/BertForPointwiseLearning.bert-base-uncased.triples.2M.pth'
    logger.info("Load model from: {}".format(model_path))
    print("Load model from: {}".format(model_path))
    model.load_state_dict(torch.load(model_path))
    """
    if args.target == 'mini_adv':
        model = pairwise_miniLM.MinitForPairwiseLearning.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model.load_state_dict(torch.load(args.model_path), strict=False)
    elif args.target == 'nb_bert':
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
    elif args.target == 'nb_bert_adv':
        model = pairwise_NB_bert.NBBERTForPairwiseLearning.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
        model.load_state_dict(torch.load(args.model_path), strict=False)
    else:
        model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    lm_model = BertForLM.from_pretrained(args.lm_model_dir)
    lm_model.to(device)
    lm_model.eval()
    for param in lm_model.parameters():
        param.requires_grad = False
    
    model, lm_model = amp.initialize([model, lm_model])
    eval_lm_model = SentenceScorer(device)

    total_docs_cnt = 0
    success_cnt = 0
    less_500_cnt = 0
    less_100_cnt = 0
    less_50_cnt = 0
    less_20_cnt = 0
    less_10_cnt = 0
    cnt = 0
    boost_rank_list = []
    q_candi_trigger_dict = dict()

    # used_qids = list(queries.keys())[:5]
    used_qids = list(queries.keys())[-100:]
    # random.shuffle(used_qids)

    for qid in tqdm(used_qids, desc="Processing"):
        torch.manual_seed(args.seed + cnt)
        torch.cuda.manual_seed_all(args.seed + cnt)
        query = queries[qid]
        best = best_query_sent[qid]
        best_score = best[0]
        best_sent = ' '.join(best[1:3])
        print('Best score: {}'.format(best_score))

        old_scores = query_scores[qid][::-1]
        if args.nature:
            trigger, new_score, trigger_cands = gen_natural_collision(
                inputs_a=query,
                inputs_b="",#best_sent,
                model=model, 
                tokenizer=tokenizer, 
                device=device, 
                lm_model=lm_model, 
                margin=best_score, 
                eval_lm_model=eval_lm_model, 
                args=args
            )
        else:
            trigger, new_score, trigger_cands = gen_aggressive_collision(
                inputs_a=query, 
                inputs_b="",#best_sent, 
                model=model, 
                tokenizer=tokenizer,
                device=device, 
                margin=best_score, 
                lm_model=lm_model, 
                args=args
            )
        
        lm_perp = eval_lm_model.perplexity(trigger)
        msg = f'Query={query}\n' \
            f'Best true sentences={best_sent}\n' \
            f'Best similarity score={best_score}\n' \
            f'Trigger={trigger}\n' \
            f'Similarity core={new_score}\n' \
            f'LM perp={lm_perp.item()}\n'

        print(msg)
        logger.info(msg)
        if args.verbose:
            logger.info('---Rank shifts for less relevant documents---')
            for did in target_q_passage[qid]:
                old_rank, old_score, label = target_q_passage[qid][did]
                new_rank = len(old_scores) - bisect.bisect_left(old_scores, new_score)
                
                total_docs_cnt += 1
                boost_rank_list.append(old_rank - new_rank)
                if old_rank > new_rank:
                    success_cnt += 1
                    if new_rank <= 500:
                        less_500_cnt += 1
                        if new_rank <= 100:
                            less_100_cnt += 1
                            if new_rank <= 50:
                                less_50_cnt += 1
                                if new_rank <= 20:
                                    less_20_cnt += 1
                                    if new_rank <= 10:
                                        less_10_cnt += 1
                print(f'Query id={qid}, Doc id={did}, '
                            f'old score={old_score:.2f}, new score={new_score:.2f}, old rank={old_rank}, new rank={new_rank}')
                logger.info(f'Query id={qid}, Doc id={did}, '
                            f'old score={old_score:.2f}, new score={new_score:.2f}, old rank={old_rank}, new rank={new_rank}')
            print('\n\n')
            logger.info('\n\n') 

        q_candi_trigger_dict[qid] = trigger
    boost_success_rate = success_cnt / (total_docs_cnt + 0.0) * 100
    less_500_rate = less_500_cnt / (total_docs_cnt + 0.0) * 100
    less_100_rate = less_100_cnt / (total_docs_cnt + 0.0) * 100
    less_50_rate = less_50_cnt / (total_docs_cnt + 0.0) * 100
    less_20_rate = less_20_cnt / (total_docs_cnt + 0.0) * 100
    less_10_rate = less_10_cnt / (total_docs_cnt + 0.0) * 100
    avg_boost_rank = np.average(boost_rank_list)
    res_str = 'Trigger Success Rate: {}\n'\
              'Trigger Average Rank: {}\n'\
              'less than 500 Rate: {}\n'\
              'less than 100 Rate: {}\n'\
              'less than 50 Rate: {}\n'\
              'less than 20 Rate: {}\n'\
              'less than 10 Rate: {}\n'.format(boost_success_rate, avg_boost_rank, less_500_rate, less_100_rate, less_50_rate, less_20_rate, less_10_rate)
    print(res_str)
    logger.info(res_str)
    """"""
    with open(prodir2 + '/saved_results/collision_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(args.target, args.regularize, args.nature, args.method , args.target_type,args.seq_len, args.beta, args.amount), 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(q_candi_trigger_dict, ensure_ascii=False))
        print("Saved in ",(prodir2 + '/saved_results/collision_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(args.target, args.regularize, args.nature, args.method , args.target_type, args.seq_len, args.beta, args.amount)))
        print('Trigger saved!')

    

def test_conllision(args):
    print('Test all triggers on imitation model...')
    # load victim models results
    target_q_passage, query_scores, best_query_sent, queries, passages_dict = prepare_data_and_scores(target_name=args.target,
                                                                                                    data_name='dl',
                                                                                                    top_k=10,
                                                                                                    least_num=5)
    if args.target == 'mini':
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model.to(device)
    elif args.target == 'mini_adv':
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model = pairwise_miniLM.MinitForPairwiseLearning.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model_path = "/data_share/chenzhuo/adversarial_data/model/miniLM_clean_2.pt"
        model.load_state_dict(torch.load(model_path), strict=False)
        model.to(device)
    elif args.target == 'large':
        tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-large-msmarco")
        model.to(device)
    else:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = pointwise_bert.BertForPointwiseLearning.from_pretrained(args.transformer_model)
        model.to(device)
        model_path = prodir + '/bert_ranker/saved_models/BertForPointwiseLearning.bert-base-uncased.triples.2M.pth'
        print("Load model from: {}".format(model_path))
        model.load_state_dict(torch.load(model_path))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    trigger_path = prodir2 + '/saved_results/collision_{}_{}_{}_{}_{}.json'.format("mini", args.regularize, args.nature, args.seq_len, args.beta)
    with open(trigger_path, 'r', encoding='utf-8') as fin:
        q_trigger_dict = json.loads(fin.readline())
    

    total_docs_cnt = 0
    success_cnt = 0
    less_500_cnt = 0
    less_100_cnt = 0
    less_50_cnt = 0
    less_20_cnt = 0
    less_10_cnt = 0
    boost_rank_list = []
    query_keys = list(q_trigger_dict.keys())
    with torch.no_grad():
        for qid in tqdm(query_keys):
            query = queries[qid]
            best = best_query_sent[qid]
            best_sent = best[1]
            old_scores = query_scores[qid][::-1]#[::-1]表示倒序输出

            trigger = q_trigger_dict[qid]
            for did in target_q_passage[qid]:
                tmp_best_new_score = -1e9
                old_rank, raw_score = target_q_passage[qid][did]
                # print("Raw score: {%.4f}".format(raw_score))
                
                # head
                # triggered_passage = trigger + ' ' + passages_dict[did]
                # triggered_passage = query + ' ' + passages_dict[did]
                # tail
                # triggered_passage = passages_dict[did] + ' ' + trigger
                half_len_passage = int(len(passages_dict[did]) / 2)
                triggered_passage = passages_dict[did][:half_len_passage] + ' ' + trigger + ' ' + passages_dict[did][half_len_passage:]#注意：中间插入trigger

                batch_encoding = tokenizer([[query, triggered_passage]], max_length=256, padding="max_length", truncation=True, return_tensors='pt')
                
                if args.target == 'mini':
                    outputs = model(**(batch_encoding.to(device)))
                    new_score = outputs.logits.squeeze().item()
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

                if tmp_best_new_score < new_score:#每个passage都更新；
                    tmp_best_new_score = new_score
                print("New high score: {:.4f}".format(new_score))
                print("Trigger: {}".format(trigger))

                new_rank = len(old_scores) - bisect.bisect_left(old_scores, tmp_best_new_score)

                total_docs_cnt += 1
                boost_rank_list.append(old_rank - new_rank)
                if old_rank > new_rank:
                    success_cnt += 1
                    if new_rank <= 500:
                        less_500_cnt += 1
                        if new_rank <= 100:
                            less_100_cnt += 1
                            if new_rank <= 50:
                                less_50_cnt += 1
                                if new_rank <= 20:
                                    less_20_cnt += 1
                                    if new_rank <= 10:
                                        less_10_cnt += 1
            
                print(f'Query id={qid}, Doc id={did}, '
                            f'old score={raw_score:.4f}, new score={tmp_best_new_score:.4f}, old rank={old_rank}, new rank={new_rank}')
            print('\n\n')
    
    boost_success_rate = success_cnt / (total_docs_cnt + 0.0) * 100
    less_500_rate = less_500_cnt / (total_docs_cnt + 0.0) * 100
    less_100_rate = less_100_cnt / (total_docs_cnt + 0.0) * 100
    less_50_rate = less_50_cnt / (total_docs_cnt + 0.0) * 100
    less_20_rate = less_20_cnt / (total_docs_cnt + 0.0) * 100
    less_10_rate = less_10_cnt / (total_docs_cnt + 0.0) * 100
    avg_boost_rank = np.average(boost_rank_list)
    print("Load trigger: {}".format(trigger_path))
    print("Collision Imitation: {}; Target: {}".format(args.target, args.experiment_name))
    res_str = 'Boost Success Rate: {}\n'\
              'Average Boost Rank: {}\n'\
              'less than 500 Rate: {}\n'\
              'less than 100 Rate: {}\n'\
              'less than 50 Rate: {}\n'\
              'less than 20 Rate: {}\n'\
              'less than 10 Rate: {}\n'.format(boost_success_rate, avg_boost_rank, less_500_rate, less_100_rate, less_50_rate, less_20_rate, less_10_rate)
    print(res_str)
    print("Collision Results:\t10\t20\t50\t100\t500\tSucc\tAvg-Boost\n")
    print("Collision Results:\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(less_10_rate, less_20_rate, less_50_rate, less_100_rate, less_500_rate, boost_success_rate, avg_boost_rank))

def test_conllision_ranking(args, tokenizer):
    print("args::", args)
    print('Test all triggers on imitation model...')
    # load victim models results
    target_q_passage, query_scores, best_query_sent, queries, passages_dict = prepare_data_and_scores(target_name=args.target,
                                                                                                    data_name='dl',
                                                                                                    top_k=10,
                                                                                                    least_num=5,
                                                                                                    target_type=args.target_type)
    if args.target == 'mini':
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model.to(device)
    elif args.target == 'mini_adv':
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model = pairwise_miniLM.MinitForPairwiseLearning.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model.load_state_dict(torch.load(args.model_path), strict=False)
        model.to(device)
    elif args.target == 'large' :
        tokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-large-msmarco")
        model.to(device)
    elif args.target == 'nb_bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
        model.to(device)
    elif args.target == 'nb_bert_adv':
        #tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = pairwise_NB_bert.NBBERTForPairwiseLearning.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
        model.load_state_dict(torch.load(args.model_path), strict=False)
        model.to(device)
    else:
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        model = pointwise_bert.BertForPointwiseLearning.from_pretrained(args.transformer_model)
        model.to(device)
        model_path = prodir + '/bert_ranker/saved_models/BertForPointwiseLearning.bert-base-uncased.triples.2M.pth'
        print("Load model from: {}".format(model_path))
        model.load_state_dict(torch.load(model_path))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    trigger_path = prodir2 + '/saved_results/collision_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(args.target, args.regularize, args.nature, args.method , args.target_type, args.seq_len, args.beta, args.amount)
    with open(trigger_path, 'r', encoding='utf-8') as fin:
        q_trigger_dict = json.loads(fin.readline())
    

    total_docs_cnt = 0
    success_cnt = 0
    less_500_cnt = 0
    less_100_cnt = 0
    less_50_cnt = 0
    less_20_cnt = 0
    less_10_cnt = 0
    boost_rank_list = []
    query_keys = list(q_trigger_dict.keys())

    with torch.no_grad():
        for qid in tqdm(query_keys):
            query = queries[qid]
            best = best_query_sent[qid]
            best_sent = best[1]
            old_scores = query_scores[qid][::-1]

            trigger = q_trigger_dict[qid]
            for did in target_q_passage[qid]:
                tmp_best_new_score = -1e9
                old_rank, raw_score, label = target_q_passage[qid][did]
                # print("Raw score: {%.4f}".format(raw_score))
                
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
                elif args.target == 'nb_bert_adv':
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

                if tmp_best_new_score < new_score:#每个passage都更新；
                    tmp_best_new_score = new_score
                print("New high score: {:.4f}".format(new_score))
                print("Trigger: {}".format(trigger))

                new_rank = len(old_scores) - bisect.bisect_left(old_scores, tmp_best_new_score)
                #rank_list.append(new_rank)

                total_docs_cnt += 1
                boost_rank_list.append(old_rank - new_rank)
                if old_rank > new_rank:
                    success_cnt += 1
                    if new_rank <= 500:
                        less_500_cnt += 1
                        if new_rank <= 100:
                            less_100_cnt += 1
                            if new_rank <= 50:
                                less_50_cnt += 1
                                if new_rank <= 20:
                                    less_20_cnt += 1
                                    if new_rank <= 10:
                                        less_10_cnt += 1
            
                print(f'Query id={qid}, Doc id={did}, '
                            f'old score={raw_score:.4f}, new score={tmp_best_new_score:.4f}, old rank={old_rank}, new rank={new_rank}')
            print('\n\n')
    
    target_q_passage_all, query_scores_all, best_query_sent_all, queries_all, passages_dict_all = prepare_data_and_scores(target_name=args.target,
                                                                                                    data_name='dl',
                                                                                                    top_k=10,
                                                                                                    least_num=0,
                                                                                                    target_type=args.target_type)
    query_keys_2 = list(q_trigger_dict.keys())
    with torch.no_grad():
        for qid in tqdm(query_keys_2):
            trigger_2 = q_trigger_dict[qid]
            for did in target_q_passage_all[qid]:
                tmp_best_new_score = -1e9
                old_rank, raw_score, label = target_q_passage_all[qid][did]
                half_len_passage = int(len(passages_dict_all[did]) / 2)
                #triggered_passage_2 = passages_dict_all[did][:half_len_passage] + ' ' + trigger_2 + ' ' + passages_dict_all[did][half_len_passage:]
                triggered_passage_2 = trigger_2 + ' ' + passages_dict_all[did]
                passages_dict_all[did] = triggered_passage_2
    examples = []
    qids = []
    pids = []
    label_s = []
    q_list = list(target_q_passage_all.keys())
    for qid in tqdm(q_list):
        d_list = list(target_q_passage_all[qid].keys())
        for did in d_list:
            label_s.append(target_q_passage_all[qid][did][2])
            examples.append([queries_all[qid] ,passages_dict_all[did]])
            qids.append(qid)
            pids.append(did)
    
    all_scores = []
    all_labels = []
    all_qids = []
    all_softmax_logits = []

    tqdm_bar = tqdm(range(0, len(label_s), args.batch_size), desc='Processing:')

    for i in tqdm_bar:
        tmp_examples = examples[i: i+args.batch_size]
        tmp_qids = qids[i: i+args.batch_size]
        tmp_pids = pids[i: i+args.batch_size]
        tmp_labels = torch.tensor(label_s[i: i + args.batch_size], dtype=torch.long)
        batch_encoding_pos = tokenizer([(e[0], e[1]) for e in tmp_examples], max_length=256, padding="max_length", truncation=True, return_tensors='pt')

        if args.target == 'distilbert-cat-margin_mse-T2-msmarco':
            input_ids = batch_encoding_pos['input_ids'].to(device)
            attention_mask = batch_encoding_pos['attention_mask'].to(device)
            scores = model(input_ids=input_ids, attention_mask=attention_mask).squeeze()

            all_scores += scores.tolist()
            all_softmax_logits += scores.tolist()
        elif args.target == 'mini':
            # input_ids = batch_encoding['input_ids'].to(device)
            # attention_mask = batch_encoding['attention_mask'].to(device)
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            outputs = model(**batch_encoding_pos.to(device))
            scores = outputs.logits.squeeze()
            all_scores += scores.tolist()
            all_softmax_logits += scores.tolist()
        elif args.target == 'mini_adv':
            pos_input_ids = batch_encoding_pos['input_ids'].to(device)
            pos_token_type_ids = batch_encoding_pos['token_type_ids'].to(device)
            pos_attention_mask = batch_encoding_pos['attention_mask'].to(device)
            neg_input_ids = batch_encoding_pos['input_ids'].to(device)
            neg_token_type_ids = batch_encoding_pos['token_type_ids'].to(device)
            neg_attention_mask = batch_encoding_pos['attention_mask'].to(device)
            true_labels = tmp_labels.to(device)
            outputs = model(
                input_ids_pos=pos_input_ids,
                attention_mask_pos=pos_attention_mask,
                token_type_ids_pos=pos_token_type_ids,
                input_ids_neg=neg_input_ids,
                attention_mask_neg=neg_attention_mask,
                token_type_ids_neg=neg_token_type_ids,
            )
            logits = outputs[0]
            all_scores += logits[:, 1].tolist()
            all_softmax_logits += torch.softmax(logits, dim=1)[:, 1].tolist()
        elif args.target == 'nb_bert_adv':
            pos_input_ids = batch_encoding_pos['input_ids'].to(device)
            pos_token_type_ids = batch_encoding_pos['token_type_ids'].to(device)
            pos_attention_mask = batch_encoding_pos['attention_mask'].to(device)
            neg_input_ids = batch_encoding_pos['input_ids'].to(device)
            neg_token_type_ids = batch_encoding_pos['token_type_ids'].to(device)
            neg_attention_mask = batch_encoding_pos['attention_mask'].to(device)
            true_labels = tmp_labels.to(device)
            outputs = model(
                input_ids_pos=pos_input_ids,
                attention_mask_pos=pos_attention_mask,
                token_type_ids_pos=pos_token_type_ids,
                input_ids_neg=neg_input_ids,
                attention_mask_neg=neg_attention_mask,
                token_type_ids_neg=neg_token_type_ids,
            )
            logits = outputs[0]
            all_scores += logits[:, 1].tolist()
            all_softmax_logits += torch.softmax(logits, dim=1)[:, 1].tolist()
        else:
            input_ids = batch_encoding_pos['input_ids'].to(device)
            token_type_ids = batch_encoding_pos['token_type_ids'].to(device)
            attention_mask = batch_encoding_pos['attention_mask'].to(device)
                
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs[0]

            all_scores += logits[:, 1].tolist()
            all_softmax_logits += torch.softmax(logits, dim=1)[:, 1].tolist()
        
        all_labels += tmp_labels
        all_qids += tmp_qids

    all_labels, _ = bert_ranker_utils.accumulate_list_by_qid(all_labels, all_qids)#得到每个qid对应的了labels列表
    all_logits, _ = bert_ranker_utils.accumulate_list_by_qid(all_scores, all_qids)
    all_softmax_logits, _ = bert_ranker_utils.accumulate_list_by_qid(all_softmax_logits, all_qids)
    res = evaluate_and_aggregate(all_logits, all_labels, ['ndcg_cut_10', 'map', 'recip_rank'])
    for metric, v in res.items():
        print("\n{} {} : {:3f}".format(args.mode, metric, v))
        #self.logger.info("{} {} : {:3f}".format(args.mode, metric, v))
    
    validation_metric = ['MAP', 'RPrec', 'MRR', 'MRR@10','NDCG', 'NDCG@10']
    all_metrics = np.zeros(len(validation_metric))
    query_cnt = 0
    for labels, logits, probs in zip(all_labels, all_logits, all_softmax_logits):
        gt = set(list(np.where(np.array(labels) > 0)[0]))
        pred_docs = np.array(probs).argsort()[::-1]
        #pred_docs_1 = np.array(probs).argmax(-1)

        all_metrics += metrics(gt, pred_docs, validation_metric)
        query_cnt += 1
    all_metrics /= query_cnt
    print("\n"+"\t".join(validation_metric))
    print("\t".join(["{:4f}".format(x) for x in all_metrics]))

    boost_success_rate = success_cnt / (total_docs_cnt + 0.0) * 100
    less_500_rate = less_500_cnt / (total_docs_cnt + 0.0) * 100
    less_100_rate = less_100_cnt / (total_docs_cnt + 0.0) * 100
    less_50_rate = less_50_cnt / (total_docs_cnt + 0.0) * 100
    less_20_rate = less_20_cnt / (total_docs_cnt + 0.0) * 100
    less_10_rate = less_10_cnt / (total_docs_cnt + 0.0) * 100
    avg_boost_rank = np.average(boost_rank_list)
    print("Load trigger: {}".format(trigger_path))
    print("Collision Imitation: {}; Target: {}".format(args.target, args.experiment_name))
    res_str = 'Boost Success Rate: {}\n'\
              'Average Boost Rank: {}\n'\
              'less than 500 Rate: {}\n'\
              'less than 100 Rate: {}\n'\
              'less than 50 Rate: {}\n'\
              'less than 20 Rate: {}\n'\
              'less than 10 Rate: {}\n'.format(boost_success_rate, avg_boost_rank, less_500_rate, less_100_rate, less_50_rate, less_20_rate, less_10_rate)
    print(res_str)
    print("Collision Results:\t10\t20\t50\t100\t500\tSucc\tAvg-Boost\n")
    print("Collision Results:\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(less_10_rate, less_20_rate, less_50_rate, less_100_rate, less_500_rate, boost_success_rate, avg_boost_rank))


if __name__ == '__main__':
    main()
