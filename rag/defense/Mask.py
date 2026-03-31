import os
import sys
from collections import defaultdict
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import torch
from defense import ranker_utils
import faiss
import pandas as pd

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

class Mask_operator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.mask_token = self.tokenizer.mask_token
        print("Nask token:", self.mask_token)

    def sampling_index_loop_nums(self, length: int,
                             mask_numbers: int, 
                             nums: int, 
                             sampling_probs: List[float] = None) -> List[int]:
        # sampling_probs: the probability of each word to be masked
        # length: the length of the sentence
        # mask_numbers: the number of words to be masked
        # nums: the number of sentences to be generated
        if sampling_probs is not None:
            assert length == len(sampling_probs)
            if sum(sampling_probs) != 1.0:
                sampling_probs = sampling_probs / sum(sampling_probs)
        mask_indexes = []
        for _ in range(nums):
            mask_indexes.append(np.random.choice(list(range(length)), mask_numbers, replace=False, p=sampling_probs).tolist()) 
        return mask_indexes

    def mask_sentence_by_indexes(self, sentence: List[str], indexes: np.ndarray, token: str, forbidden: List[str]=None) -> str:
        tmp_sentence = sentence.copy()
        for index in indexes:
            tmp_sentence[index] = token
        if forbidden is not None:
            for index in forbidden:
                tmp_sentence[index] = sentence[index]
        return ' '.join(tmp_sentence)
    
    def mask_sentence(self,
                  sentence: str, 
                  rate: float,
                  token: str, 
                  nums: int = 1, 
                  return_indexes: bool = False, 
                  forbidden: List[int] = None,
                  random_probs: List[float] = None, 
                  mask_numbers: int = None, 
                  min_keep: int = 2) -> List[str]:
        # sentence: the sentence to be masked
        # rate: the rate of words to be masked
        # token: the token to replace the masked words
        # nums: the number of sentences to be generated; amount of copies
        # return_indexes: whether to return the indexes of masked words
        # forbidden: the indexes of words that cannot be masked
        # random_probs: the probability of each word to be masked
        # min_keep: the minimum number of words to be kept
        
        # str --> List[str]
        sentence_in_list = sentence.split()
        length = len(sentence_in_list)

        if mask_numbers is None:
            mask_numbers = round(length * rate)
        else:
            mask_numbers = mask_numbers
        if length - mask_numbers < min_keep:
            mask_numbers = length - min_keep if length - min_keep >= 0 else 0

        mask_indexes = self.sampling_index_loop_nums(length, mask_numbers, nums, random_probs)
        tmp_sentences = []
        remain_indexes = []
        for indexes in mask_indexes:
            tmp_sentence = self.mask_sentence_by_indexes(sentence_in_list, indexes, token, forbidden)
            tmp_sentences.append(tmp_sentence)
        if return_indexes:
            # get other indices except for the masked ones
            for indexes in mask_indexes:
                remain_indexes.append(list(set(range(length)) - set(indexes)))
            return tmp_sentences, mask_indexes, remain_indexes
        else:
            return tmp_sentences

    def product_score(self, model, query, candidates, k=10 ,measure = "dot"):
        if not isinstance(candidates, list):
            query_emb = model.encode_(query)[0].cpu().numpy()
            doc_embs = model.encode_(candidates).cpu().numpy()
        else:
            query_emb = model.encode([query])[0]
            if type(query_emb) != np.ndarray:
                query_emb = query_emb.cpu().numpy()
                doc_embs = doc_embs.cpu().numpy()
        
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
        
        return dis_score, ind[0]

    def product_score_for_list(self, model, query, candidates, k=10 ,measure = "dot"):
        if not isinstance(candidates, list):
            query_emb = model.encode_(query)[0].cpu().numpy()
            doc_embs = model.encode_(candidates).cpu().numpy()
        else:
            query_emb = model.encode([query])[0]
            doc_embs = model.encode(candidates)
            if type(query_emb) != np.ndarray:
                query_emb = query_emb.cpu().numpy()
                doc_embs = doc_embs.cpu().numpy()
        
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
        return dis_score, ind[0]

    def masked_ranker(self, rate, nums, examples, collection_str, model, output_dir = "/msmarco/train/runs_for_defense", max_seq_len = 256, mode = "infer", run_id = "contriever_procon_mask"):#examples:[[query, passage]]
        with torch.no_grad():
            model.eval()

            all_logits = []
            all_flat_labels = []
            all_softmax_logits = []
            all_qids = []
            all_pids = []
            cnt = 0
            num = 0
            for q in tqdm(list(examples.keys())[:]):#q,qid
                num+=1
                tmp_query = q
                print(num, " - ", tmp_query)
                passages = []
                tmp_pid = []
                text_to_did = {}

                for d in examples[q].keys():
                    query_ = examples[q][d][0]
                    if tmp_query != query_:
                        print(tmp_query, 'vs', query_)
                        raise ValueError("Unmatched Query!")
                    passages.append(collection_str[d])
                    tmp_pid.append(d)
                    text_to_did[collection_str[d]] = d

                cnt += 1
                for j in range(len(passages)):
                    tmp_passage = passages[j]

                    masked_samples = self.mask_sentence(tmp_passage, rate, token=self.mask_token, nums=nums)
                    
                    # masked_instances = []
                    # for masked_sample in masked_samples:
                    #     masked_instances.append((tmp_query, masked_sample))

                    example_logits = []
                    example_softmax_logits = []

                    sim_score , _ = self.product_score_for_list(model, tmp_query, masked_samples, k = nums)

                    example_logits += sim_score#logits[:, 1].tolist()
                    example_softmax_logits += torch.softmax(torch.tensor(np.array([sim_score])), dim=1)#[:, 1].tolist()
                    # calculate mean logits
                    example_mean_logit = np.mean(example_logits)
                    example_mean_softmax_logit = np.mean(np.array(example_softmax_logits[0]))
            
                    all_logits.append(example_mean_logit)
                    all_softmax_logits.append(example_mean_softmax_logit)
                    all_qids.append(q)
                    all_pids.append(tmp_pid[j])

            # accumulates per query
            all_logits, _ = ranker_utils.accumulate_list_by_qid(all_logits, all_qids)
            all_softmax_logits, _ = ranker_utils.accumulate_list_by_qid(all_softmax_logits, all_qids)
            all_pids, all_qids = ranker_utils.accumulate_list_by_qid(all_pids, all_qids)
        

        if mode not in ['dev', 'test']:
            # For TREC eval
            runs_list = []
            ranked_dic = {}
            for scores, qids, pids in zip(all_logits, all_qids, all_pids):
                sorted_idx = np.array(scores).argsort()[::-1]
                sorted_scores = np.array(scores)[sorted_idx]
                sorted_qids = np.array(qids)[sorted_idx]
                sorted_pids = np.array(pids)[sorted_idx]
                ranked_dic[sorted_qids[0]] = [examples[sorted_qids[0]][p][1] for p in sorted_pids]
                for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
                    runs_list.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'Dense-mono'))
            runs_df = pd.DataFrame(runs_list, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
            # runs_df.to_csv(
            #     output_dir + '/runs.' + run_id + '.' + mode  + '.' + str(
            #         max_seq_len) + '.csv', sep='\t', index=False, header=False)
            return ranked_dic
            

    
