from langchain_community.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import LLMChain,HuggingFacePipeline,PromptTemplate
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pickle as pkl
from transformers import AutoTokenizer,  pipeline
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
from LocalEmbedding import localEmbedding
from scipy.stats import spearmanr
from imitation_agreement import top_n_overlap_sim, rbo_score
from evaluate import cal_NDCG
# from metrics import evaluate_and_aggregate
import label_smoothing
import random
import re
import json
from bert_ranker_utils import accumulate_list_by_qid_and_pid, accumulate_list_by_pid, accumulate_list_by_qid_2_dic, accumulate_list_by_qid
from condenser import sim_score, sim_score_for_passage_list

current_file = __file__
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)

msmarco_collection_path = grandparent_dir+'//msmarco/msmarco_passage/collection_queries/collection.tsv'
msmarco_queries_path = grandparent_dir+'/msmarco/msmarco_passage/collection_queries/queries.dev.tsv'
msmarco_qrels_path = grandparent_dir+'/msmarco/msmarco_passage/qrels.dev.tsv'
imitation_data_path = grandparent_dir+'/msmarco/samples/4_up.pkl'
run_bm25 = grandparent_dir+'/msmarco/msmarco_passage/sampled_set/run_bm25.tsv'

def comparison_main():
    topk = 10
    #embedding
    BGE_NAME = grandparent_dir+"/model_hub/bge-large-en-v1.5"
    CON_NAME = grandparent_dir+'/model_hub/condenser/co-condenser-marco/'
    model_kwargs = {'device': 'cuda',
                    }#'add_pooling_layer': False,'output_hidden_states': True,
    encode_kwargs = {'normalize_embeddings': True}

    device = 'cuda'
    embedding_model = localEmbedding(
            CON_NAME,
            device
        )
    
    # load query
    query_df = pd.read_csv(msmarco_queries_path, names=['qid','query_string'], sep='\t')
    query_df['qid'] = query_df['qid'].astype(str)
    queries_str = query_df.set_index('qid').to_dict()['query_string']
    with open(imitation_data_path, "rb") as f:
        data = pkl.load(f)#{query: {1:[]; 2:[]}}
    f.close()
    qid_list = list(data.keys())
# 
    model_path = grandparent_dir+"/model_hub/Qwen-1.5/Qwen1.5-14B-Chat"
    device = torch.device("cuda:0")
    from qwen_LLM import Qwen_LLM
    llm = Qwen_LLM(mode_name_or_path = model_path)

    #PROMPT
    template_ranks = """
        The following are passages about Question #{question}.

        Passages: {context}.

        Rank these passages based on their relevance to the question. Do not segment the passages. You MUST include all the passages. For eaxmple:
        (2) > (3) > (1) > ...

        """
    template = """
        Use the following pieces of retrieved context to answer the question. Keep the answer concise.:
    
        Context: {context}.

        Question: {question}. 
        """
    
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    prompt = ChatPromptTemplate.from_template(template)

    data_rm = {}
    data_rank = {}
    spearman_sum = []
    rbo_list = []
    overlap_list = []
    ndcg_rm_list = []
    ndcg_llm_list = []


    for qid in tqdm(qid_list[:]):
        query = queries_str[str(qid)]
        print("QUERY:", query)
        texts = ['('+ str(i+1) +') '+data[qid][1][i][2] for i in range(len(data[qid][1]))]
        related_num = len(texts)
        texts.extend(['('+ str(i+1+related_num) +') '+data[qid][0][i][2] for i in range(len(data[qid][0]))])
        text_labels = {i+1:1 for i in range(len(data[qid][1]))}
        text_labels.update({i+1+related_num:0 for i in range(len(data[qid][0]))})

        texts_no_number = [data[qid][1][i][2] for i in range(len(data[qid][1]))]+[data[qid][0][i][2] for i in range(len(data[qid][0]))]
        texts_dic = {'('+ str(i+1) +')':data[qid][1][i][2] for i in range(len(data[qid][1]))}
        texts_dic.update({'('+ str(i+1+related_num) +')':data[qid][0][i][2] for i in range(len(data[qid][0]))})
        text_2_id = {data[qid][1][i][2]:i+1 for i in range(len(data[qid][1]))}
        text_2_id.update({data[qid][0][i][2]:i+1+related_num for i in range(len(data[qid][0]))})
        text_doc_id = {'('+ str(i+1) +')':data[qid][1][i][0] for i in range(len(data[qid][1]))}
        text_doc_id.update({'('+ str(i+1+related_num) +')':data[qid][0][i][0] for i in range(len(data[qid][0]))})

        db =  FAISS.from_texts(texts_no_number, embedding_model)
        search_result = db.similarity_search(query, k=topk)
        retrieval_id_rank = [text_2_id[t.page_content] for t in search_result]

        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": topk})
        print("**********************************")

        template_rank_7b = """Question:""" + query + """ 
Passages:\n\n"""+"\n\n".join(texts)+"\n\nRank all the passages below based on their relevance to the question concisely, just output the passages' ranking in the form of: (number) > (number) > (number) > (number) > (number) > (number) > (number) > (number) > (number) > (number); Forget about the context provided and include passage (1)(2)(3)(4)(5)(6)(7)(8)(9)(10)."
        template_rank = """Question:""" + query + """ 
Passages:\n\n"""+"\n\n".join(texts)+"""\n\nRank the passages above based on their relevance to the question! Output the ranking in the form of: (number of the most relevant passage) > (number of relevant passage) > ...... > (number of irrelevant passage) > (number of the most irrelevant passage), for example:(8) > (4) > (6) > (2) > (7) > (10) > (5).
        The output MUST have 10 passage nuumbers."""

        print("**********************************")
        # format_docs(serach_result)
        def format_docs_no_order(docs):
            # result = "\n\n".join(doc.page_content for doc in docs)
            result = [docs[i].page_content for i in range(len(docs))]
            result_t = [list(texts_dic.keys())[list(texts_dic.values()).index(t)] for t in result]
            # result_t = [str(text_2_id[t]) for t in result]
            # result = ['['+ docs[i].page_content+ ']' for i in range(len(docs))]
            result_t = " > ".join(result_t)
            print("MID:", result_t)
            print("**********************************")
            return result
        
        rag_chain = (
            {"context": retriever | format_docs_no_order, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
        rank_output = rag_chain.invoke(template_rank)
        rank_list = [t.strip() for t in rank_output.split(">")]
        if len(rank_list)<10:
            print(qid, " DATA ABANDOM!")
            continue
        else:
            rank_list_2 = [int(rank_list[i][1:-1]) for i in range(len(rank_list))]
            print("LLM:", rank_list_2)
            print("RM:", retrieval_id_rank)
            spearman, _ = spearmanr(rank_list_2, retrieval_id_rank)
            rbo = rbo_score(retrieval_id_rank, rank_list_2, p=0.7, max_depth=10)
            top_overlap = top_n_overlap_sim(retrieval_id_rank, rank_list_2, topn=10)
            labels_rm = [text_labels[t] for t in retrieval_id_rank]
            ndcg_rm = cal_NDCG(list(range(len(labels_rm), 0, -1)), labels_rm)
            ndcg_llm = cal_NDCG(list(range(len(labels_rm), 0, -1)), [text_labels[t] for t in rank_list_2])
            print("NDCG:", ndcg_rm, ndcg_llm)
            print("SPEARMAN:", spearman)
            print("rbo:", rbo, top_overlap)
            rbo_list.append(rbo)
            overlap_list.append(top_overlap)
            spearman_sum.append(spearman)
            ndcg_rm_list.append(ndcg_rm)
            ndcg_llm_list.append(ndcg_llm)
        
        data_rm[query] = {text_doc_id['('+ str(retrieval_id_rank[i]) +')']:i for i in range(len(retrieval_id_rank))}
        data_rank[query] = {text_doc_id[rank_list[i]]:i for i in range(len(rank_list))}
    
    print('spearman:', sum(spearman_sum)/len(spearman_sum))
    print('RBO:', sum(rbo_list)/len(rbo_list))
    print('overlap:', sum(overlap_list)/len(overlap_list))
    print('NDCG_RM:', sum(ndcg_rm_list)/len(ndcg_rm_list))
    print('NDCG_LLM:', sum(ndcg_llm_list)/len(ndcg_llm_list))

def test_pairwise_with_DR(pkl_path, measure = "dot"):
    #preparation
    from Condenser_model import CondenserForPairwiseModel_msmarco
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    target_model = CondenserForPairwiseModel_msmarco.from_pretrained(grandparent_dir+"/model_hub/condenser/msmarco-bert-co-condensor")
    target_model.to('cuda')

    #data
    with open(pkl_path, "rb") as f:
        data = pkl.load(f)
    f.close()
    print("LOADING queries...")
    query_df = pd.read_csv(msmarco_queries_path, names=['qid','query_string'], sep='\t')
    query_df['qid'] = query_df['qid'].astype(str)
    queries_str = query_df.set_index('qid').to_dict()['query_string']

    all_score = []
    for qid in tqdm(data):
        query = queries_str[str(qid)]
        for pair in data[qid]:
            pos = pair[0]
            neg = pair[1]
            # pos_dis = L2_score(target_model, query, pos)
            # neg_dis = L2_score(target_model, query, neg)
            dis, ind = sim_score_for_passage_list(target_model, query, [pos, neg], 2, measure=measure)
            pos_dis , neg_dis = dis[0], dis[1]
            # print("dis: pos:", pos_dis, neg_dis)
            dis_gap = neg_dis - pos_dis
            all_score.append(dis_gap)
    print("DATA: ", pkl_path)
    if measure == "dot":
        print("Good rate: " , len([t for t in all_score if t < 0])/len(all_score))
    elif measure == "L2":
        print("Good rate: " , len([t for t in all_score if t > 0])/len(all_score))
    

def msmarco_load_and_sample(bound = 4 ,sample_num = 20, path = grandparent_dir+"/msmarco/samples/4_up_top100.pkl"):
    msmarco_qrels_path = grandparent_dir+'/msmarco/msmarco_passage/qrels.dev.tsv'
    if os.path.exists(path):
        with open(path, 'rb') as f:
                print("Loading instances from {}".format(path))
                examples = pkl.load(f)
                labels = pkl.load(f)
                qids = pkl.load(f)
                pids = pkl.load(f)
        f.close()
        labels_q_p = accumulate_list_by_qid_and_pid(labels, pids, qids)
        pid_2_text = accumulate_list_by_pid(examples, pids)
        qid_2_text = accumulate_list_by_qid_2_dic(examples, qids)

        new_data = {}
        q_ids = list(labels_q_p.keys())
        for i in range(len(q_ids)):
            samples = sorted(labels_q_p[q_ids[i]].items(),key = lambda d:d[1],reverse=True)
            start = 0
            for j in range(len(samples)):
                if samples[j][1] == 0:
                    start = j
                    break
            if start < int(sample_num/2):
                sample_cut = start
            else:
                sample_cut = int(sample_num/2)
            samples_ = samples[:sample_cut]
            samples_.extend(samples[start:start+(sample_num - sample_cut)])
            # new_data[q_ids[i]] = {t[0]:t[1] for t in samples}
            random.shuffle(samples_)
            new_data[q_ids[i]] = samples_
        
        return new_data, pid_2_text, qid_2_text
    else:
        # load doc_id=pid to string
        msmarco_collection_path = grandparent_dir+"/msmarco/msmarco_passage/collection_queries/collection.tsv"
        print("LOADING PASSAGE.....")
        collection_df = pd.read_csv(msmarco_collection_path, sep='\t', names=['docid', 'document_string'])
        collection_df['docid'] = collection_df['docid'].astype(str)
        collection_str = collection_df.set_index('docid').to_dict()['document_string']
        # load query
        print("LOADING QUERY.....")
        msmarco_queries_path = grandparent_dir+"/msmarco/msmarco_passage/collection_queries/queries.dev.tsv"
        query_df = pd.read_csv(msmarco_queries_path, names=['qid','query_string'], sep='\t')
        query_df['qid'] = query_df['qid'].astype(str)
        queries_str = query_df.set_index('qid').to_dict()['query_string']
        qrels_df = pd.read_csv(msmarco_qrels_path, delim_whitespace= True,names=['qid', 'iter', 'docid', 'relevance'])
        temt = ''
        terms = []
        nums = 0
        have_dic = {}
        tempt_list = []
        for i in tqdm(range(len(qrels_df))):
            if qrels_df.loc[i, 'qid'] == temt:
                nums+=1
                tempt_list.append(qrels_df.loc[i, 'docid'])
            else:
                if nums>=bound and temt not in terms:
                    terms.append(temt)
                    print(temt,":", nums)
                    have_dic[temt] = tempt_list
                temt = qrels_df.loc[i, 'qid']
                tempt_list=[qrels_df.loc[i, 'docid']]
                nums = 1
        print("T",len(terms))
        examples = []#
        labels = []
        qids = []
        pids = []
        limit = 100*len(terms)

        # target_q = [[t, queries_str[str(t)]] for t in terms]
        for i in tqdm(range(len(qrels_df))):
            if qrels_df.loc[i, 'qid'] in terms:
                qids.append(qrels_df.loc[i, 'qid'])
                labels.append(qrels_df.loc[i, 'relevance'])
                pids.append(qrels_df.loc[i, 'docid'])
                examples.append((queries_str[str(qrels_df.loc[i, 'qid'])], collection_str[str(qrels_df.loc[i, 'docid'])]))
                print(len(labels))
            else:
                if len(labels) >= limit:
                    continue
                elif random.choice([True, False]):
                    ran_q = random.choice(terms)
                    if qrels_df.loc[i, 'docid'] in have_dic[ran_q]:
                        continue

                    qids.append(ran_q)
                    labels.append(0)
                    pids.append(qrels_df.loc[i, 'docid'])
                    examples.append((queries_str[str(ran_q)], collection_str[str(qrels_df.loc[i, 'docid'])]))
        print(len(labels))
        with open(path, "wb") as f:
            pkl.dump(examples, f)
            pkl.dump(labels, f)
            pkl.dump(qids, f)
            pkl.dump(pids, f)
        f.close()
        print(path," SAVED!")
        return None, None, None

def msmarco_run_bm25_load():
    # load doc_id=pid to string
    path = grandparent_dir+"/msmarco/samples/from_run_bm25_origin_no_messy_code.pkl"
    if not os.path.exists(path):
        print("LOADING colletion...")
        collection_df = pd.read_csv(msmarco_collection_path, sep='\t', names=['docid', 'document_string'])#, encoding='ISO-8859-1'
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
            # csv_reader = csv.reader(f)
            for line in tqdm(f):
                line = ''.join(line)
                line = line.strip().split('\t')
                if len(line) != 5:
                    print(line)
                qrels_df.append(line)
    
        data = {}
        pid_2_text, qid_2_text = {}, {}
        for i in tqdm(range(1,len(qrels_df))):
            qid = qrels_df[i][0]
            did = qrels_df[i][1]
            query = qrels_df[i][2]
            passage = qrels_df[i][3].encode('raw_unicode_escape').decode('utf8')
            label = qrels_df[i][4]
            if qid not in data:
                data[qid] = [(did, label)]
            else:
                data[qid].append((did, label))
            if qid not in qid_2_text:
                qid_2_text[qid] = query
            if did not in pid_2_text:
                pid_2_text[did] = passage
        for q in data.keys():
            print(len(data[q]))
        print(len(data.keys()))
        with open(path, "wb") as f:
            pkl.dump(data, f)
            pkl.dump(pid_2_text, f)
            pkl.dump(qid_2_text, f)
        f.close()
        print(path," SAVED!")
        return None, None, None
    else:
        with open(path, 'rb') as f:
            print("Loading instances from {}".format(path))
            data = pkl.load(f)
            pid_2_text = pkl.load(f)
            qid_2_text = pkl.load(f)
        f.close()
        return data, pid_2_text, qid_2_text


def extract_dict(s) -> list:
    """Extract all valid dicts from a string.
    
    Args:
        s (str): A string possibly containing dicts.
    
    Returns:
        A list containing all valid dicts.
    
    """
    results = []
    s_ = ' '.join(s.split('\n')).strip()
    exp = re.compile(r'(\{.*?\})')

    for i in exp.findall(s_):
        try:
            if 'context' not in json.loads(i):
                results.append({'answer': [],
                        'context':[]})
            results.append(json.loads(i))
        except json.decoder.JSONDecodeError:
            results.append({'answer': [],
                        'context':[]})
    #     try:
    #         results.append(json.loads(i))        
    #     except json.JSONDecodeError:
    #         pass
    if len(results) == 0:
        results.append({'answer': [],
                        'context':[]})
    if 'context' not in results[-1]:
        results.append({'answer': [],
                        'context':[]})
    return results[-1]

        


if __name__ == '__main__':
    # comparison_main()
    # msmarco_load_and_sample()
    # test_between_dr_and_surrogate()
    msmarco_run_bm25_load()


