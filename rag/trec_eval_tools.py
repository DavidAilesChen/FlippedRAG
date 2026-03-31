import sys
import sys
import os

# current_file = __file__
# parent_dir = os.path.dirname(current_file)
# grandparent_dir = os.path.dirname(parent_dir)

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
grandparent_dir = os.path.dirname(curdir)

from trectools import TrecQrel, TrecRun, TrecEval

data_dir = grandparent_dir+'/mamarco'
ms_dev_small_qrels_file = data_dir + '/msmarco_passage/collection_queries/qrels.dev.small.tsv'
ms_dev1000_small_qrel_file = data_dir + '/msmarco_passage/subsmall/msmarco_ans_small/qrels.dev.small.tsv'
dl_2019_qrels_file = grandparent_dir+'/opinion_pro/trec_dl_2019/2019qrels-pass.txt'
dl_2019_qrels_file_binary = grandparent_dir+'/opinion_pro/trec_dl_2019/trec_dl_2019_qrels.tsv'
mb2014_qrels_file = data_dir + '/trec_mb_2014/qrels.mb2014.txt'

ms_runs = grandparent_dir + '/results/runs/runs.RankingBERT.public.bert.msmarco.Tue_Aug_29.eval_shrink_bm25_dev1000_0_none.csv'

dl_runs = grandparent_dir+"/msmarco/train/runs_for_revision/runs.DR_QWEN3_4b_on_trec_dl_2019.target_test_dot.csv"

mb_runs = curdir + '/results/runs/runs.bert-base-uncased.pairwise.triples.mb2014.csv'


def transform_run_to_runs(run_file, runs_file):
    qid_list, did_list, rank_list = [], [], []
    with open(run_file, 'r') as f:
        for line in f:
            qid, did, rank = line.strip().split('\t')
            qid_list.append(qid)
            did_list.append(did)
            rank_list.append(int(rank))
    with open(runs_file, 'w') as f:
        for qid, did, rank in zip(qid_list, did_list, rank_list):
            f.write('\t'.join([qid, "Q0", did, str(rank), str(1000 - rank), 'dev']))
            f.write('\n')


def eval_msmarco():
    # For MSMARCO Passage
    run1 = TrecRun(ms_runs)
    qrels = TrecQrel(ms_dev_small_qrels_file)
    trec_eval = TrecEval(run1, qrels)
    run_mrr_10 = trec_eval.get_reciprocal_rank(depth=10)
    run_ndcg_10 = trec_eval.get_ndcg(depth=10)
    print("MRR@10: {}\n".format(run_mrr_10))
    print("nDCG@10: {}\n".format(run_ndcg_10))


def eval_dl2019():
    # For TREC DL 2019
    run1 = TrecRun(dl_runs)
    qrels_binary = TrecQrel(dl_2019_qrels_file_binary)

    trec_eval = TrecEval(run1, qrels_binary)
    run_mrr_10 = trec_eval.get_reciprocal_rank(depth=10)

    print("MRR@10: {}\n".format(run_mrr_10))

    qrels_grade = TrecQrel(dl_2019_qrels_file)
    trec_eval2 = TrecEval(run1, qrels_grade)
    run_ndcg_10 = trec_eval2.get_ndcg(depth=10)
    print("NDCG@10: {}\n".format(run_ndcg_10))


def eval_mb2014():
    run1 = TrecRun(mb_runs)
    qrels = TrecQrel(mb2014_qrels_file)
    trec_eval = TrecEval(run1, qrels)
    run_p30 = trec_eval.get_precision(depth=30)
    run_ap = trec_eval.get_map(depth=1000)
    print("P@30: {}\n".format(run_p30))
    print("AP: {}\n".format(run_ap))


if __name__ == "__main__":
    # transform_run_to_runs(ms_bm25_run, ms_bm25_runs)
    # eval_mb2014()
    print(dl_runs)
    eval_dl2019()
    #eval_msmarco()


