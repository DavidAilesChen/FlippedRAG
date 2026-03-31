import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

# from metrics import evaluate_and_aggregate, set_recall, metrics
import bert_ranker_utils
import label_smoothing
from sklearn.metrics import f1_score,recall_score,precision_score, classification_report

import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import pickle as pkl
import functools
import operator
from copy import deepcopy
# from apex import amp
from Imitation_Encoder import MinitForPairwiseLearning, MiniForPairwiseClassfy, NBBERTForPairwiseClassfy, BertForPairwiseLearning
from imitation_agreement import top_n_overlap_dic_sim, rbo_dict_score

current_file = __file__
parent_dir = os.path.dirname(current_file)
grandparent_dir = os.path.dirname(parent_dir)
model_dir = os.path.dirname(os.path.dirname(grandparent_dir))+"/model_hub"

class Trainer(object):
    """
    Parent class for all ranking models.
    """
    def __init__(self, model, data_class, tokenizer, args, model_path, start_model_path=None, writer_train=None, run_id='BERT',
                 monitor_metric='ndcg_cut_10', validation_metric=None, writer_eval=None, logger=None):
        # Config dict format
        # self.validation_metric = config["validation_metric"]
        # self.validate_every_epochs = config["validate_every_epochs"]
        # self.validate_every_steps = config["validate_every_steps"]
        # self.num_validation_batches = config["num_validation_batches"]
        # self.num_epochs = config["num_epochs"]
        # self.lr = config["lr"]
        # self.batch_size = config["batch_size"]
        # self.num_training_instances= config["num_training_instances"]
        # self.max_grad_norm = config["max_grad_norm"]
        # self.data_name = config["data_name"]
        # self.accumulation_steps = config["accumulation_steps"]
        # self.warmup_portion = config["warmup_portion"]
        
        # ArgumentParser format
        self.random_seed = args.seed
        self.num_epochs = args.num_epochs
        self.num_training_instances= args.num_training_instances
        self.validate_every_epochs = args.validate_every_epochs
        self.validate_every_steps = args.validate_every_steps
        self.num_validation_batches = args.num_validation_batches
        self.batch_size_train = args.train_batch_size
        self.batch_size_eval = args.val_batch_size
        self.sample_num = args.sample_data
        self.use_dev_triple = args.use_dev_triple
        self.pseudo_final = args.pseudo_final

        self.max_seq_len = args.max_seq_len
        self.lr = args.lr_train
        self.max_grad_norm = args.max_grad_norm
        self.accumulation_steps = args.accumulation_steps
        self.warmup_portion = args.warmup_portion
        self.perturb_iter = args.perturb_iter
        
        self.transformer_model = args.transformer_model
        self.loss_function = args.loss_function
        self.smoothing = args.smoothing

        self.data_class = data_class   
        self.tokenizer = tokenizer     
        self.monitor_metric = monitor_metric
        self.validation_metric = validation_metric
        self.model_name = args.model_name

        self.num_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device {}".format(self.device))
        print("Num GPU {}".format(self.num_gpu))

        self.model = model.to(self.device)
        
        self.writer_train = writer_train
        self.logger = logger
        
        self.model_path = model_path
        self.output_dir = grandparent_dir+'/msmarco/train/'
        self.run_id = run_id

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        if self.num_gpu > 1:
            devices = [v for v in range(self.num_gpu)]
            self.model = nn.DataParallel(self.model, device_ids=devices)
            self.optimizer = nn.DataParallel(self.optimizer, device_ids=devices)
        if start_model_path is not None:
            self.model = self.load_state(self.model, start_model_path, self.num_gpu)

    def load_state(self, model, model_path, gpu_num):
        if gpu_num > 1:
            model.module.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path))
        print("{} loaded!".format(model_path))
        self.logger.info("{} loaded!".format(model_path))
        return model
    
    def load_model(self):
        
        if self.model_name == 'pairwise-minilm-ranker':
            # model = MinitForPairwiseLearning.from_pretrained(self.transformer_model,loss_function=self.loss_function, smoothing=self.smoothing)
            model = MiniForPairwiseClassfy.from_pretrained(self.transformer_model,loss_function=self.loss_function, smoothing=self.smoothing)
        # elif self.model_name == 'pairwise-NBbert-ranker':
            # model = pairwise_NB_bert.NBBERTForPairwiseLearning.from_pretrained(self.transformer_model,
                        # loss_function=self.loss_function, smoothing=self.smoothing)
        elif self.model_name == 'pairwise-minilm-learner':
            model = MinitForPairwiseLearning.from_pretrained(self.transformer_model,loss_function=self.loss_function, smoothing=self.smoothing)
        elif self.model_name == 'pairwise-nbbert-ranker':
            # model = NBBERTForPairwiseClassfy.from_pretrained("nboost/pt-bert-base-uncased-msmarco",loss_function=self.loss_function, smoothing=self.smoothing)
            model = BertForPairwiseLearning.from_pretrained(model_dir+"/nboost_pt-bert-base-uncased-msmarco",loss_function=self.loss_function, smoothing=self.smoothing)#nboost/pt-bert-base-uncased-msmarco
        else:
            raise ValueError("{} model class is not exist!".format(self.model_name))

        model.to(self.device)
        # model = amp.initialize(model, opt_level='O1')
        if self.num_gpu > 1:
            devices = [v for v in range(self.num_gpu)]
            model = nn.DataParallel(model, device_ids=devices)
            model.module.load_state_dict(torch.load(self.model_path))
        else:
            model.load_state_dict(torch.load(self.model_path))
        """
        model = self.model"""
        return model
    
    def train_ranker_pairwise(self, mode='train'):
        max_val = 0
        save_best = True
        global_step = 0

        data_generator = self.data_class.data_generator_pairwise_triple
        
        for epoch in range(self.num_epochs):
            print("Train the models for epoch {} with batch size {}\n".format(epoch, self.batch_size_train))
            self.logger.info("Train the models for epoch {} with batch size {}".format(epoch, self.batch_size_train))
            self.model.train()

            epoch_instance = 0
            epoch_step = 0
            early_stop_cnt = 0
        
            for batch_encoding_pos, batch_encoding_neg, tmp_labels in data_generator(#mode=mode,
                                                                                    #epoch_sample_num=self.sample_num,
                                                                                    #random_seed=self.random_seed+epoch,
                                                                                    batch_size=self.batch_size_train,
                                                                                    #max_seq_len=self.max_seq_len
                                                                                    ):
                pos_input_ids = batch_encoding_pos['input_ids'].to(self.device)
                pos_token_type_ids = batch_encoding_pos['token_type_ids'].to(self.device)
                pos_attention_mask = batch_encoding_pos['attention_mask'].to(self.device)
                neg_input_ids = batch_encoding_neg['input_ids'].to(self.device)
                neg_token_type_ids = batch_encoding_neg['token_type_ids'].to(self.device)
                neg_attention_mask = batch_encoding_neg['attention_mask'].to(self.device)
                true_labels = tmp_labels.to(self.device)
                outputs = self.model(
                    input_ids_pos=pos_input_ids,
                    attention_mask_pos=pos_attention_mask,
                    token_type_ids_pos=pos_token_type_ids,
                    input_ids_neg=neg_input_ids,
                    attention_mask_neg=neg_attention_mask,
                    token_type_ids_neg=neg_token_type_ids,
                    labels=true_labels
                )
                loss = outputs[0]

                if self.num_gpu > 1:
                    loss = loss.mean()
                    print("multi-loss_sum:", loss)
                # logits_dif = outputs[-1]
                # pre_labels = torch.argmax(logits_dif, dim=-1).tolist()
                # precision = precision_score(tmp_labels.tolist(), pre_labels)
                # recall = recall_score(tmp_labels.tolist(), pre_labels)
                # macro_f1 = f1_score(tmp_labels.tolist(), pre_labels, average='macro')
                # micro_f1 = f1_score(tmp_labels.tolist(), pre_labels, average='micro')

                # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    # scaled_loss.backward()
                loss.backward()

                self.writer_train.add_scalar('train_loss', loss, global_step)
                # self.writer_train.add_scalar('p', precision, global_step)
                # self.writer_train.add_scalar('r', recall, global_step)
                # self.writer_train.add_scalar('pos_neg_mac_f1', macro_f1, global_step)
                # self.writer_train.add_scalar('pos_neg_mic_f1', micro_f1, global_step)

                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)
                # nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_grad_norm)
                
                self.optimizer.step()
                # self.scheduler.step()
                self.optimizer.zero_grad()
                global_step += 1
                epoch_step += 1
                epoch_instance += pos_input_ids.shape[0]

                if self.num_training_instances != -1 and epoch_instance >= self.num_training_instances:
                    print("Reached num_training_instances of {} ({} batches). Early stopping.\n".format(self.num_training_instances, epoch_step))
                    self.logger.info("Reached num_training_instances of {} ({} batches). Early stopping.".format(self.num_training_instances, epoch_step))
                    break

                #logging for steps
                is_validation_step = (self.validate_every_steps > 0 and global_step % self.validate_every_steps == 0)
                if is_validation_step:
                    with torch.no_grad():
                        # eval_loss, res_dict = self.eval_ranker(mode='dev', model=self.model)
                        # res_array = self.dev_pairwise(global_step, mode='dev', model=self.model)
                        if self.use_dev_triple and 'pseudo' not in mode:
                            mac_f1 = self.dev_triple_pairwise(global_step, mode='dev_triple', model=self.model)
                        else:
                            mac_f1 = 0

                        # if res_array[self.validation_metric.index(self.monitor_metric)] + mac_f1 > max_val and save_best:
                        #     max_val = res_array[self.validation_metric.index(self.monitor_metric)] + mac_f1
                        if  mac_f1 > max_val and save_best:#res_array[self.monitor_metric] +
                            # max_val = res_array[self.monitor_metric] + mac_f1
                            max_val =  mac_f1
                            """"""
                            if self.num_gpu > 1:
                                torch.save(self.model.module.state_dict(), self.model_path)
                            else:
                                torch.save(self.model.state_dict(), self.model_path)
                            self.logger.info("Saved !")
                            print("\nSaved !")
                            early_stop_cnt = 0
                        else:
                            early_stop_cnt += 1
                
                if early_stop_cnt > 2:
                    print("early stop this epoch")
                    self.logger.info("early stop this epoch")
                    break

            is_validation_epoch = (self.validate_every_epochs > 0 and (epoch % self.validate_every_epochs == 0))
            if is_validation_epoch:
                with torch.no_grad():
                    # eval_loss, res_dict = self.eval_ranker(mode='dev', model=self.model)
                    # res_array = self.dev_pairwise(global_step ,mode='dev', model=self.model)
                    if self.use_dev_triple and 'pseudo' not in mode:
                        mac_f1 = self.dev_triple_pairwise(global_step, mode='dev_triple', model=self.model)
                    else:
                        mac_f1 = 0
                    # if res_array[self.validation_metric.index(self.monitor_metric)] + mac_f1 > max_val and save_best:
                    #     max_val = res_array[self.validation_metric.index(self.monitor_metric)] + mac_f1
                    if  mac_f1 > max_val and save_best:#res_array[self.monitor_metric] +
                        # max_val = res_array[self.monitor_metric] + mac_f1
                        max_val = mac_f1
                        # self.model.save_pretrained(self.model_path)
                        """"""
                        if self.num_gpu > 1:
                            torch.save(self.model.module.state_dict(), self.model_path)
                        else:
                            torch.save(self.model.state_dict(), self.model_path)
                        self.logger.info("Saved !")
                        print("\nSaved !")
        """
        TEST
        """
        with torch.no_grad():
            model = self.load_model()
            self.dev_pairwise(step=global_step, mode='test', model=model)
            if self.pseudo_final and 'pseudo' not in mode:
                self.pseudo_pairwise(mode='test', model=model)
            # self.dev_pairwise(mode='eval_full_dev1000', model=self.model)

    def dev_pairwise(self, step, mode='dev', model=None):
        model.eval()

        all_logits = []
        all_labels = []
        all_smooth_labels = []
        all_softmax_logits = []
        all_qids = []
        all_pids = []
        cnt = 0

        
        for batch_encoding, tmp_labels, tmp_true_labels, tmp_qids, tmp_pids in self.data_class.data_generator_ranking_dev(mode=mode,
                                                                                batch_size=self.batch_size_eval,
                                                                                max_seq_len=self.max_seq_len):
            cnt += 1
            pos_input_ids = batch_encoding['input_ids'].to(self.device)
            pos_token_type_ids = batch_encoding['token_type_ids'].to(self.device)
            pos_attention_mask = batch_encoding['attention_mask'].to(self.device)
            neg_input_ids = batch_encoding['input_ids'].to(self.device)
            neg_token_type_ids = batch_encoding['token_type_ids'].to(self.device)
            neg_attention_mask = batch_encoding['attention_mask'].to(self.device)
            true_labels = tmp_true_labels.to(self.device)
            outputs = model(
                input_ids_pos=pos_input_ids,
                attention_mask_pos=pos_attention_mask,
                token_type_ids_pos=pos_token_type_ids,
                input_ids_neg=neg_input_ids,
                attention_mask_neg=neg_attention_mask,
                token_type_ids_neg=neg_token_type_ids,
                #labels=true_labels
            )
            if self.loss_function == "label-smoothing-cross-entropy" and self.model_name == "pairwise-minilm-ranker":
                # loss_func = label_smoothing.LabelSmoothingCrossEntropy(self.smoothing)
                loss_func = nn.MSELoss()
            elif self.loss_function == "label-smoothing-cross-entropy" and self.model_name == "pairwise-minilm-learner":
                loss_func = label_smoothing.LabelSmoothingCrossEntropy(self.smoothing)
            elif self.loss_function == "label-smoothing-cross-entropy" and self.model_name == "pairwise-nbbert-ranker":
                loss_func = label_smoothing.LabelSmoothingCrossEntropy(self.smoothing)
            else:
                loss_func = nn.CrossEntropyLoss(size_average=False, reduce=True) 
            if outputs[0].shape[1] >= 2:
                val_loss = loss_func(outputs[0], tmp_labels.to(self.device).view(-1))
                if mode in ['dev', 'dev_cocondenser']:
                    self.writer_train.add_scalar('val_loss', val_loss, step)
                elif mode in ['test']:
                    pass
            else:
                val_loss = loss_func(outputs[0], tmp_labels.to(self.device).view(-1))
                if mode in ['dev', 'dev_cocondenser']:
                    self.writer_train.add_scalar('val_loss', val_loss, step)
                elif mode in ['test']:
                    pass

            logits = outputs[0]
            all_labels += true_labels.int().tolist() # this is required sbecause of the weak supervision
            all_smooth_labels += tmp_labels.cpu()

            if outputs[0].shape[1] >= 2:
                all_logits += logits[:, 1].tolist()
                all_softmax_logits += [torch.argmax(t).cpu() for t in logits]
            else:
                all_logits += logits[:, 0].tolist()
                all_softmax_logits += map(lambda x: 1 if x[0] >= 0.5 else 0, logits)
            # all_softmax_logits += torch.softmax(logits, dim=1)[:, 1].tolist()
            all_qids += tmp_qids
            all_pids += tmp_pids

            if self.num_validation_batches!=-1 and cnt > self.num_validation_batches and (mode == 'dev' or mode == 'dev_cocondenser'):
                break
        
        #Classification Evaluation
        # print(all_true_labels, all_softmax_logits)
        # presicion_value = precision_score(all_smooth_labels, all_softmax_logits, average='binary')
        # recall_value = recall_score(all_smooth_labels, all_softmax_logits, average='binary')
        # f1_score_value = f1_score(all_smooth_labels, all_softmax_logits, average='binary')
        #accumulates per query
        all_labels, _ = bert_ranker_utils.accumulate_list_by_qid(all_labels, all_qids)
        logit_pid_qid_dict = bert_ranker_utils.accumulate_list_by_qid_and_pid(all_logits, all_pids, all_qids)
        all_logits, _ = bert_ranker_utils.accumulate_list_by_qid(all_logits, all_qids)
        # all_softmax_logits, _ = bert_ranker_utils.accumulate_list_by_qid(all_softmax_logits, all_qids)
        all_pids, all_qids = bert_ranker_utils.accumulate_list_by_qid(all_pids, all_qids)

        res = evaluate_and_aggregate(all_logits, all_labels, ['ndcg_cut_10', 'map', 'recip_rank'])
        truth_ndcg = None

        # res['set_recall'] = set_recall_rate
        # res['inter'] = overlap
        # res['rbo'] = rbo
      
        for metric, v in res.items():
            print("\n{} {} : {:3f}".format(mode, metric, v))
            """
            self.logger.info("{} {} : {:3f}".format(mode, metric, v))
            if mode not in ["test"]:
                self.writer_train.add_scalar(metric, v, step)"""

        return res
    
    def dev_triple_pairwise(self, step, mode='dev_triple', model=None):
        model.eval()
        all_pre_labels = []
        all_labels = []
        cnt = 0


        for batch_encoding_pos, batch_encoding_neg, tmp_labels in self.data_class.data_generator_pairwise_dev_triple(mode=mode,
                                                                                            batch_size=self.batch_size_eval,
                                                                                            max_seq_len=self.max_seq_len):
            pos_input_ids = batch_encoding_pos['input_ids'].to(self.device)
            pos_token_type_ids = batch_encoding_pos['token_type_ids'].to(self.device)
            pos_attention_mask = batch_encoding_pos['attention_mask'].to(self.device)
            neg_input_ids = batch_encoding_neg['input_ids'].to(self.device)
            neg_token_type_ids = batch_encoding_neg['token_type_ids'].to(self.device)
            neg_attention_mask = batch_encoding_neg['attention_mask'].to(self.device)
            true_labels = tmp_labels.to(self.device)
            outputs = model(
                input_ids_pos=pos_input_ids,
                attention_mask_pos=pos_attention_mask,
                token_type_ids_pos=pos_token_type_ids,
                input_ids_neg=neg_input_ids,
                attention_mask_neg=neg_attention_mask,
                token_type_ids_neg=neg_token_type_ids,
                labels=true_labels
            )

            if self.loss_function == "label-smoothing-cross-entropy" and self.model_name == "pairwise-minilm-ranker":
                # loss_func = label_smoothing.LabelSmoothingCrossEntropy(self.smoothing)
                loss_func = nn.MSELoss()
            elif self.loss_function == "label-smoothing-cross-entropy" and self.model_name == "pairwise-minilm-learner":
                loss_func = label_smoothing.LabelSmoothingCrossEntropy(self.smoothing)
            elif self.loss_function == "label-smoothing-cross-entropy" and self.model_name == "pairwise-nbbert-ranker":
                loss_func = label_smoothing.LabelSmoothingCrossEntropy(self.smoothing)
            else:
                loss_func = nn.CrossEntropyLoss(size_average=False, reduce=True) 
            if outputs[-1].shape[1] >= 2:
                val_loss = loss_func(outputs[-1], tmp_labels.to(self.device).view(-1))
                if mode in ['dev', 'dev_triple']:
                    self.writer_train.add_scalar('triple_val_loss', val_loss, step)
                elif mode in ['test']:
                    pass
            else:
                val_loss = loss_func(outputs[-1], tmp_labels.to(self.device).view(-1))
                if mode in ['dev', 'dev_triple']:
                    self.writer_train.add_scalar('triple_val_loss', val_loss, step)
                elif mode in ['test']:
                    pass

            logits_dif = outputs[-1]
            pre_labels = torch.argmax(logits_dif, dim=-1).tolist()
            all_pre_labels += pre_labels
            all_labels += tmp_labels.tolist()

            cnt += 1

            if self.num_validation_batches!=-1 and cnt > self.num_validation_batches and (mode == 'dev' or mode == 'dev_triple'):#self.num_validation_batches
                break
        
        macro_f1 = f1_score(all_labels, all_pre_labels, average='macro')
        print(classification_report(all_labels, all_pre_labels))
        self.logger.info(classification_report(all_labels, all_pre_labels))
        self.logger.info("Pairwise triples {}: {}".format(mode, macro_f1))
        return macro_f1

