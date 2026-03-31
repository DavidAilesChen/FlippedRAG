import sys
import os
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
"""
import json
import torch
import logging
import pickle as pkl
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizerFast, BertConfig, AutoModelForSequenceClassification, BertForNextSentencePrediction
from transformers import AutoTokenizer, AutoModel
from bert_ranker.models import pairwise_miniLM, pairwise_NB_bert, pairwise_bert, pairwise_NB_bert_classifier 
from bert_ranker.models.modeling import RankingBERT_Train, RankingBERT_Pairwise, RankBertForPairwise
from opinion_reverse.dense_retrieval.Condenser_model import CondenserForPairwiseModel, CondenserForPairwiseModel_msmarco
from opinion_reverse.dense_retrieval.DenseRetrieval_model import ContrieverForPairwiseModel, ANCEForPairwiseModel, QwenEmbeddingForPairwiseModel
from preprocess.dataset_stance import Stance_Dataset
import torch
import torch.nn as nn
import collections
import argparse
from ir.scorer import SentenceScorer
# from apex import amp
#from nltk.corpus import stopwords
from ranker import opinion_ranking, dense_dual_encoder_ranking, eval_ranking
# from opinion_process import attack_rank
from attack_on_opinion import attack_opinion_with_triggers
from ir.scorer import SentenceScorer
from ir.bert_models import BertForLM, BGEForLM
from evaluate import avg_rank_boost, topk_proportion, cal_NDCG , relabel_polarity, topk_mutual_score, recall_score, topk_proportion_to_length, rbo_score
current_file = __file__
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)
model_dir = os.path.dirname(os.path.dirname(grandparent_dir))+"/model_hub"
surrogate_model_dir = os.path.dirname(grandparent_dir)+"/msmarco/train/models_for_QWEN"

BERT_LM_MODEL_DIR = model_dir+'/bert/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

parser = argparse.ArgumentParser('Pytorch')
parser.add_argument("--data_name", default='fnc', type=str)
parser.add_argument("--max_seq_len", default=256, type=int)
parser.add_argument("--target", type=str, default='bge', help='test on what model')#nb_bert,bge
parser.add_argument('--eval_on_other', default=True ,type=bool, help='Decide whether evaluate triggers on other models')
parser.add_argument("--polarity", type=int, default=0, help='target polarity.')
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--nsp_model", default="/bert-base-uncased", type=str, required=False, help="Bert model to use (cross-encoder/ms-marco-MiniLM-L-12-v2,bert-base-uncased).")
parser.add_argument("--transformer_model", default="NBbert-for-adversarial", type=str, required=False, help="Bert model to use (cross-encoder/ms-marco-MiniLM-L-12-v2,bert-base-uncased).")
parser.add_argument('--stemp', type=float, default=1.0, help='temperature of softmax')
parser.add_argument('--lr', type=float, default=0.1, help='optimization step size')
parser.add_argument('--max_iter', type=int, default=2, help='maximum iteraiton')
parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
parser.add_argument('--pat', action='store_true', help='Use PAT to decrease perplexity')
parser.add_argument('--fp16', default=True, action='store_true', help='fp16')
parser.add_argument('--patience_limit', type=int, default=2, help="Patience for early stopping.")
parser.add_argument("--seed", default=42, type=str, help="random seed")
parser.add_argument("--num_sims", default=300, type=int, help="number of PAT augmentation words.")
parser.add_argument("--lambda_1", default=0.1, type=float, help="lambda1")
parser.add_argument("--lambda_2", default=0.7, type=float, help="lambda2")
parser.add_argument('--min_len', type=int, default=5, help='Min sequence length')
parser.add_argument("--beta", default=0.0, type=float, help="Coefficient for language model loss.")
parser.add_argument("--amount", default=0, type=int, help="adv_Data amount.")
parser.add_argument('--save', action='store_true', help='Save collision to file')
parser.add_argument('--verbose', action='store_true', default=True,  help='Print every iteration')
parser.add_argument("--lm_model_dir", default=BERT_LM_MODEL_DIR, type=str, help="Path to pre-trained language model")
parser.add_argument('--perturb_iter', type=int, default=3, help='PPLM iteration')
parser.add_argument("--kl_scale", default=0.0, type=float, help="KL divergence coefficient")
parser.add_argument("--topk", default=50, type=int, help="Top k sampling for beam search")
parser.add_argument("--num_beams", default=10, type=int, help="Number of beams")
parser.add_argument("--num_filters", default=100, type=int, help="Number of num_filters words to be filtered")
parser.add_argument('--nature', action='store_true', help='Nature collision')
parser.add_argument('--nsp',  help='whether to use next sentence prediction.')
parser.add_argument('--regularize', action='store_true', help='Use regularize to decrease perplexity')
parser.add_argument('--batch_split', action='store_false', help='To split the data into batches')
parser.add_argument('--query_plus', action='store_true', help='Query+++')
parser.add_argument('--eval_model', default="ance", type=str)

args = parser.parse_args()


def main():
    args.query_plus = True

    if args.target == "bge":
        tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    else:
        tokenizer = BertTokenizerFast.from_pretrained(model_dir+"/bert-base-uncased")#bert-base-uncased
    if args.target == 'nb_bert_adv':
        # model_rank = pairwise_NB_bert.NBBERTForPairwiseLearning.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
        # target_model = pairwise_NB_bert.NBBERTForPairwiseLearning.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
        target_model = pairwise_NB_bert_classifier.NBBERTForPairwiseClassfy.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
        # model_rank.load_state_dict(torch.load(args.model_path), strict=False)
        # target_model.load_state_dict(torch.load(args.model_path), strict=False)
    elif args.target == 'nb_bert':
        if args.pat:
            # model_rank = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
            # model_rank = pairwise_NB_bert.NBBERTForPairwiseLearning.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
            # target_model = pairwise_NB_bert.NBBERTForPairwiseLearning.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
            target_model = pairwise_bert.BertForPairwiseLearning.from_pretrained(model_dir+"/nboost_pt-bert-base-uncased-msmarco")#nboost/pt-bert-base-uncased-msmarco
            # target_model = pairwise_NB_bert_classifier.NBBERTForPairwiseClassfy.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
            nbbert_path = surrogate_model_dir+"/NBbert_epoch4_dropout_QWEN_black_bm25_origin_nomessycode_sample3x10-50fromnbrank_top60_dot_500q_batch256_tripledev_4e5.pt"
            target_model.load_state_dict(torch.load(nbbert_path), strict=False)
            print("LOADED ",nbbert_path)
        elif args.query_plus:
            target_model = AutoModelForSequenceClassification.from_pretrained(model_dir+"/nboost_pt-bert-base-uncased-msmarco")
        else:
            # model_rank = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
            target_model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
    elif args.target == "mini_im":#IMITATION
        if args.pat or args.nature or args.query_plus:
            # target_model = pairwise_miniLM.MinitForPairwiseLearning.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
            target_model = pairwise_miniLM.MiniForPairwiseClassfy.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
            mini_im_path = surrogate_model_dir+"/Mini_epoch10_dropout_runbm25sample_dot_allq_batch128_2e6.pt"
            target_model.load_state_dict(torch.load(mini_im_path), strict=False)
            print("LOADED ", mini_im_path)
    elif args.target == "mini":
        target_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    elif args.target == 'bge':
        if args.pat:
            pass
        else:
            # target_model = BGE.BGE_single.from_pretrained('BAAI/bge-m3', device=device)
            target_model = BGE.BGE_single('BAAI/bge-m3', device=device)
    elif args.target == 'condenser':
        if args.pat:
            target_model = CondenserForPairwiseModel_msmarco.from_pretrained(model_dir+'/msmarco-bert-co-condensor/')#"Luyu/co-condenser-marco"
            # gen_model = pairwise_NB_bert_classifier.NBBERTForPairwiseClassfy.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
        else:
            target_model = CondenserForPairwiseModel_msmarco.from_pretrained(model_dir+'/msmarco-bert-co-condensor/')
            # gen_model = pairwise_NB_bert_classifier.NBBERTForPairwiseClassfy.from_pretrained("nboost/pt-bert-base-uncased-msmarco")

    target_model.to(device)

    if args.target != "bge":
        lm_model = BertForLM.from_pretrained(args.lm_model_dir)
        lm_model.to(device)
        lm_model.eval()
    else:
        # lm_model = BGEForLM.from_pretrained('BAAI/bge-m3', device)
        lm_model = BGEForLM('BAAI/bge-m3', device)
        lm_model.to(device)
        lm_model.eval()

    for param in lm_model.parameters():
        param.requires_grad = False
    eval_lm_model = SentenceScorer(device)
    if args.pat:
        nsp_model = BertForNextSentencePrediction.from_pretrained(args.nsp_model)
        nsp_model.to(device)
        nsp_model.eval()
        for param in nsp_model.parameters():
            param.requires_grad = False
    else:
        nsp_model = None

    if args.eval_on_other:
        # another_model = ContrieverForPairwiseModel.from_pretrained(model_dir+"/contriever_msmarco")
        # print("EVAL ON Contriever...")
        another_model = QwenEmbeddingForPairwiseModel(model_dir+"/Qwen3-Embedding-4B")
        print("EVAL ON ", args.eval_model)

        another_model.eval()
        another_model.to(device)
    
    dataset = Stance_Dataset(tokenizer=tokenizer)
    if args.data_name == "fnc":
        data, articles, origin_stance_dic= dataset.load_detected_fnc_stance()#[Body ID,Stance]
    elif args.data_name == "procon":
        data, origin_stance_dic = dataset.load_procon_data(num=6)
    queries_list = list(data.keys())

    result_list = []
    rbo_list = []
    eval_result_list = []
    eval_success = []
    eval_total = []
    eval_boost_rank = []
    passage_nums = []
    trigger_saves = {}
    passage_num = 0

    for i in tqdm([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, ]):
        print("NUMBER:", i)
        # origin_stance_list = data[queries_list[i]]
        origin_stance_list = origin_stance_dic[queries_list[i]]#[id, label, None, passage, query]
        print("LEN:", len(origin_stance_list))
        query = queries_list[i]
        print("QUERY:", query)
        # continue
        # for j in range(len(origin_stance_list)):
            # origin_stance_list[j].extend([articles[origin_stance_list[j][0]], queries_list[i]])#[body_id, stance, detected_stance, article body,query]
        if args.target == "bge" or args.target == 'condenser':
            sorted_index, sorted_label, sorted_data, sorted_true_label, sorted_logits = dense_dual_encoder_ranking(origin_stance_list, target_model, tokenizer, max_seq_len=args.max_seq_len, args=args, device=device)#sorted_data:[origin_id, label, none, passage, query]
        else:
            sorted_index, sorted_label, sorted_data, sorted_true_label, sorted_logits = opinion_ranking(origin_stance_list, target_model, tokenizer, max_seq_len=args.max_seq_len, args=args, device=device, device_cpu=device_cpu)
        if args.eval_on_other:
            sorted_index_eval, sorted_label_eval, sorted_data_eval, sorted_true_label_eval, sorted_logits_eval = eval_ranking(origin_stance_list, another_model, tokenizer, max_seq_len=args.max_seq_len, args=args, device=device)
            print("##Sequence similarity：", rbo_score(sorted_index, sorted_index_eval, p=0.7, max_depth=len(sorted_index)))
            rbo_list.append(rbo_score(sorted_index, sorted_index_eval, p=0.7, max_depth=len(sorted_index)))
        # continue
        print("sorted_labels:",sorted_label)
        print("true_labels:", sorted_true_label)
        
        
        # target:passage, query, rank ,id
        if args.eval_on_other:
            print("EVAL ON OTHER MODEL...")
            attack_target = [[sorted_data[i][3:], i, sorted_data[i][0] ] for i in range(len(sorted_label)) if sorted_label[i] == args.polarity]
            #[[passage, query], rank, id]
            target_info = [[i, sorted_logits] for i in range(len(sorted_label)) if sorted_label[i] == args.polarity]# and i < 8 and i > 0; and i<int(len(sorted_label)/2)
        else:
            attack_target = [[sorted_data[i][3:], i, sorted_data[i][0] ] for i in range(len(sorted_label)) if sorted_label[i] == args.polarity and i<int(len(sorted_label)/2)]#i>int(len(sorted_label)-5); and i<int(len(sorted_label)/2)
            target_info = [[i, sorted_logits] for i in range(len(sorted_label)) if sorted_label[i] == args.polarity and i<int(len(sorted_label)/2)]# and i < 8 and i > 0; and i<int(len(sorted_label)/2)
        # attack
        anchor_p_list = []
        if args.pat:
            if args.eval_on_other:
                anchor_p_list = [sorted_data[0][3]]*len(attack_target)
            else:
                anchor_p_list = [sorted_data[0][3]]*len(attack_target)
        target_p_list_triggers, triggers = attack_opinion_with_triggers(target_model, lm_model, eval_lm_model, nsp_model, tokenizer, query, attack_target, anchor_p_list, device=device, args=args)
        # Rerank
        if args.data_name == "fnc" or args.data_name == "procon":
            for i in range(len(sorted_data)):
                for t in target_p_list_triggers:#target_p_list_triggers:[[triggered passage, query], id]
                    if i == t[1]:
                        if args.eval_on_other:
                            sorted_data[i] = [sorted_data[i][0],sorted_data[i][1], sorted_data[i][2],t[0][0],t[0][1]]#[id, stance, None, triggered passage, query]
                        else:
                            sorted_data[i] = [sorted_data[i][0],sorted_data[i][1], sorted_data[i][2],t[0][0],t[0][1]]
        #SAVE data
        if args.save:
            trigger_saves[query] = sorted_data

        if args.target == "bge" or args.target == 'condenser':
            sorted_index_attacked, sorted_label_attacked, sorted_data_attacked, sorted_true_label_attacked, sorted_logits_attacked = dense_dual_encoder_ranking(sorted_data, target_model, tokenizer, max_seq_len=args.max_seq_len, args=args, device=device, target_info=target_info)
        else:
            sorted_index_attacked, sorted_label_attacked, sorted_data_attacked, sorted_true_label_attacked, sorted_logits_attacked = opinion_ranking(sorted_data, target_model, tokenizer, max_seq_len=args.max_seq_len, args=args, device=device, device_cpu=device_cpu)

        if args.eval_on_other:
            sorted_index_eval_attacked, sorted_label_eval_attacked, sorted_data_eval_attacked, sorted_true_label_eval_attacked, sorted_logits_eval_attacked = eval_ranking(sorted_data,another_model, tokenizer, max_seq_len=args.max_seq_len, args=args, device=device, target_info=target_info)

        # Evaluation
        if args.eval_on_other:
            print("EVAL_SINGLE:")
            result_eval_other = topk_proportion(sorted_label_eval, sorted_label_eval_attacked, args)
            eval_result_list.append(result_eval_other)
            AVG_BOOST_eval, SUM_BOOST_eval, NUM_eval = avg_rank_boost(sorted_label_eval, sorted_label_eval_attacked, args)
            eval_boost_rank.append(SUM_BOOST_eval)
            passage_nums.append(len(sorted_label_eval))
            # cal attack success rate
            success_num = 0
            for t in attack_target:
                id = t[-1]#id
                ori_rank = -1
                new_rank = -1
                for m in range(len(sorted_data_eval)):
                    if sorted_data_eval[m][0] == id:
                        ori_rank = m
                for n in range(len(sorted_data_eval_attacked)):
                    if sorted_data_eval_attacked[n][0] == id:
                        new_rank = n
                if new_rank<ori_rank:
                    success_num+=1
            eval_success.append(success_num)
            eval_total.append(len(attack_target))
            print("EVAL:SUCCESS:", eval_success, eval_total)
        
        result = topk_proportion(sorted_label, sorted_label_attacked, args)
        # result.update(topk_proportion_to_length(sorted_label, sorted_label_attacked, args, [int(len(sorted_label)/2)]))
        result.update(topk_mutual_score(sorted_label, sorted_label_attacked, args, [3, 1/2]))
        result.update(recall_score(sorted_label, sorted_label_attacked, args))
        
        AVG_BOOST, SUM_BOOST, NUM = avg_rank_boost(sorted_label, sorted_label_attacked, args)

        relabel = relabel_polarity(args.polarity, sorted_label)
        ndcg_ori = cal_NDCG(sorted_logits , relabel)
        relabel_atk = relabel_polarity(args.polarity, sorted_label_attacked)
        ndcg_atk = cal_NDCG(sorted_logits_attacked, relabel_atk)
        print("Original NDCG:", ndcg_ori)
        print("Manipulated NDCG:", ndcg_atk)
        ndcg_differ = (ndcg_atk - ndcg_ori)/(1-ndcg_ori)

        print("AVG BOOST RANK:", AVG_BOOST)
        result["AVG BOOST RANK"] = AVG_BOOST
        result['SUM_BOOST'] = SUM_BOOST
        result['NUM'] = NUM
        result['NDCG_DIFFER'] = ndcg_differ
        result_list.append({query:result})
        passage_num += len(origin_stance_list)
    #SAVE
    save_path = os.path.dirname(grandparent_dir)+"/opinion_pro/triggers/for_QWEN/pat_one_passages_from_epoch4_QWEN_black_bm25_origin_nomessycode_sample3x10-50fromnbrank_top60_dot_500q_batch256_tripledev_4e5_16_45"
    if args.save:
        with open(save_path, "wb") as f:
            pkl.dump(trigger_saves, f)
        f.close()
        print("SAVED!", save_path)
    AVG_ORI_TOP5 = 0
    AVG_ORI_TOP10 = 0
    AVG_TOP5_RATE = 0
    AVG_TOP10_RATE = 0
    AVG_TOP_PROPORTION = 0
    AVG_5BOOST_RATE = 0
    AVG_10BOOST_RATE = 0
    AVG_TOP_PROPORTION_BOOST = 0
    AVG_BOOST_RANK = 0
    AVG_3MUTUAL_SCORE = 0
    AVG_6MUTUAL_SCORE = 0
    SUM_NDCG_DIFFER = 0
    SUM_NUM = 0
    AVG_RECALL = 0
    for i in range(len(result_list)):
        AVG_ORI_TOP5 += result_list[i][list(result_list[i].keys())[0]]['before-top3']
        AVG_ORI_TOP10 += result_list[i][list(result_list[i].keys())[0]]['before-top6']
        AVG_TOP5_RATE += result_list[i][list(result_list[i].keys())[0]]['later-top3']
        AVG_TOP10_RATE += result_list[i][list(result_list[i].keys())[0]]['later-top6']
        AVG_BOOST_RANK += result_list[i][list(result_list[i].keys())[0]]['SUM_BOOST']
        AVG_5BOOST_RATE += result_list[i][list(result_list[i].keys())[0]]['top3 ']
        AVG_10BOOST_RATE += result_list[i][list(result_list[i].keys())[0]]['top6 boost']
        SUM_NDCG_DIFFER += result_list[i][list(result_list[i].keys())[0]]['NDCG_DIFFER']
        SUM_NUM += result_list[i][list(result_list[i].keys())[0]]['NUM']
        AVG_3MUTUAL_SCORE += result_list[i][list(result_list[i].keys())[0]]['top3 mutual boost score']
        AVG_6MUTUAL_SCORE += result_list[i][list(result_list[i].keys())[0]]['top1/2 mutual boost score']
        AVG_RECALL += result_list[i][list(result_list[i].keys())[0]]['recall boost score']

    SUM_EVAL_ORI_TOP3 = 0
    SUM_EVAL_ORI_TOP6 = 0
    SUM_EVAL_ATK_TOP3 = 0
    SUM_EVAL_ATK_TOP6 = 0
    if args.eval_on_other:
        print("RBO:", sum(rbo_list)/len(rbo_list))
        for i in range(len(eval_result_list)):
            SUM_EVAL_ORI_TOP3 += eval_result_list[i]['before-top3']
            SUM_EVAL_ORI_TOP6 += eval_result_list[i]['before-top6']
            SUM_EVAL_ATK_TOP3 += eval_result_list[i]['later-top3']
            SUM_EVAL_ATK_TOP6 += eval_result_list[i]['later-top6']
        AVG_EVAL_ORI_TOP3 = SUM_EVAL_ORI_TOP3/len(eval_result_list)
        AVG_EVAL_ORI_TOP6 = SUM_EVAL_ORI_TOP6/len(eval_result_list)
        AVG_EVAL_TOP3_RATE = SUM_EVAL_ATK_TOP3/len(eval_result_list)
        AVG_EVAL_TOP6_RATE = SUM_EVAL_ATK_TOP6/len(eval_result_list)
        print("EVAL_ON_OTHER:")
        print("ORI top3:", AVG_EVAL_ORI_TOP3, " top6:", AVG_EVAL_ORI_TOP6)
        print("ATK top3:", AVG_EVAL_TOP3_RATE, " top6:", AVG_EVAL_TOP6_RATE)
        print("top3_gap:", AVG_EVAL_TOP3_RATE-AVG_EVAL_ORI_TOP3, " top6_gap:", AVG_EVAL_TOP6_RATE-AVG_EVAL_ORI_TOP6)
        print("avg_ASR:", sum(eval_success)/sum(eval_total))
        print("boost_rank:", sum(eval_boost_rank)/len(eval_boost_rank))
        print("boost_rate:", sum([eval_boost_rank[i]/passage_nums[i] for i in range(len(eval_boost_rank))])/len(eval_boost_rank))
        print("__________________")

    AVG_ORI_TOP5 = AVG_ORI_TOP5/len(result_list)
    AVG_ORI_TOP10 = AVG_ORI_TOP10/len(result_list)
    AVG_TOP5_RATE = AVG_TOP5_RATE/len(result_list)
    AVG_TOP10_RATE = AVG_TOP10_RATE/len(result_list)
    AVG_BOOST_RANK = AVG_BOOST_RANK/SUM_NUM
    AVG_NDCG_DIFFER = SUM_NDCG_DIFFER/len(result_list)
    AVG_5BOOST_RATE = AVG_5BOOST_RATE/len(result_list)
    AVG_10BOOST_RATE = AVG_10BOOST_RATE/len(result_list)
    # AVG_TOP_PROPORTION_BOOST = AVG_TOP_PROPORTION_BOOST/len(result_list)
    AVG_3MUTUAL_SCORE = AVG_3MUTUAL_SCORE/len(result_list)
    AVG_6MUTUAL_SCORE = AVG_6MUTUAL_SCORE/len(result_list)
    AVG_RECALL = AVG_RECALL/len(result_list)
    print("Average passage number:", passage_num/len(result_list))
    print("AVG_BOOST_RANK:", AVG_BOOST_RANK/(passage_num/len(result_list)))
    print("AVG_3BOOST_RATE:", AVG_5BOOST_RATE)
    print("AVG_6BOOST_RATE:", AVG_10BOOST_RATE)
    print("AVG_1/2MUTUAL_SCORE:", AVG_6MUTUAL_SCORE)
    print("AVG_RECALL_BOOST:", AVG_RECALL)
    print("MUTAUL_NDCG_DIFFER:", AVG_NDCG_DIFFER)
    

if __name__ == '__main__':
    main()
# python opinion_manipulate.py --stemp=0.4 --lr=0.1 --pat --num_beams=30 --topk=100  --data_name=procon --target=nb_bert --seq_len=10 --max_seq_len=256
