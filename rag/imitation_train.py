import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '4'
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
import torch
from torch import cuda
import json
import pandas as pd
import numpy as np
# from apex import amp
import logging
from transformers import (
    AutoTokenizer,
    AutoModel,
    HfArgumentParser,
    BertForNextSentencePrediction,
    BertTokenizerFast,
)
from imitation_training_args import TrainingArguments
from Imitation_Encoder import MinitForPairwiseLearning, MiniForPairwiseClassfy, NBBERTForPairwiseClassfy, BertForPairwiseLearning

from dataset import RAG_Dataset
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer

current_file = __file__
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)
model_dir = os.path.dirname(os.path.dirname(grandparent_dir))+"/model_hub"

device = 'cuda' if cuda.is_available() else 'cpu'
print()

def main():
    parser = HfArgumentParser((TrainingArguments))
    training_args = parser.parse_args_into_dataclasses()[0]
    print(training_args)

    #SURROGATE MODEL
    if training_args.model_name_or_path == "MiniLM":
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model = MinitForPairwiseLearning.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        # model = MiniForPairwiseClassfy.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        # model = NBBERTForPairwiseClassfy.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
        # model = model.to(device)
    elif training_args.model_name_or_path == "NBBert":
        tokenizer = AutoTokenizer.from_pretrained(model_dir+"/nboost_pt-bert-base-uncased-msmarco")
        model = BertForPairwiseLearning.from_pretrained(model_dir+"/nboost_pt-bert-base-uncased-msmarco")

    #DATASET
    data_object = RAG_Dataset(tokenizer=tokenizer, exist=True)

    #LOGGER
    tensorboard_dir = grandparent_dir+"/msmarco/train/tensorlog_NBep6_blackbox_dropout_QWEN_bm25_origin_nomessycode_sample3x10-50fromnbrank_top60_dot_500q_batch256_tripledev_5e5/"
    if training_args.mode == 'train' or training_args.mode == 'embed_adv_train':
        writer_train = SummaryWriter(tensorboard_dir + 'train')
    else:
        writer_train = None

    logger = logging.getLogger("Pytorch")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    log_path = grandparent_dir+"/msmarco/train/logs/logger.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Runing with configurations: {}'.format(json.dumps(training_args.__dict__, indent=4)))

    trainer = Trainer(
        model=model,
        data_class=data_object,
        tokenizer=tokenizer,
        model_path=grandparent_dir+"/msmarco/train/models_for_QWEN/NBbert_epoch6_dropout_QWEN_black_bm25_origin_nomessycode_sample3x10-50fromnbrank_top60_dot_500q_batch256_tripledev_5e5.pt",
        start_model_path=None,
        validation_metric=['ndcg_cut_10', 'map', 'recip_rank'],
        monitor_metric='ndcg_cut_10',#set_recall',
        args=training_args,
        writer_train=writer_train,
        #run_id=training_args.run_id,
        logger=logger
        )
    
    if training_args.mode in ['train']:
        trainer.train_ranker_pairwise(mode=training_args.mode)
    else:
        trainer.dev_pairwise(step=0 ,mode=training_args.mode, model=trainer.model)

if __name__ == '__main__':
    main()

# nohup python imitation_train.py --num_epochs=4 --train_batch_size=256 --lr_train=3e-5 --model_name_or_path=NBBert --model_name=pairwise-nbbert-ranker --num_validation_batches=512 --use_dev_triple > nohup_NBep4_dropout_runbm25_black+csample3x10-50_top60_dot_503q_batch256_tripledev_3e5_samesourcerank.out