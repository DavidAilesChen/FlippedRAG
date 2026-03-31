import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)
import torch
import numpy as np
from transformers import BertPreTrainedModel, BertModel, AutoModel, AutoTokenizer, AutoModelForPreTraining, AutoModelForSequenceClassification
# from bert_ranker.losses import label_smoothing
from torch import nn
from condenser import condenser_loss, condenser_encode, compute_similarity, norm_sim
model_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))+"/model_hub"

class CondenserForPairwiseModel(BertPreTrainedModel):
    def __init__(self, config, loss_function="mse", smoothing=0.1) -> None:
        super().__init__(config)
        self.condenser = AutoModel.from_pretrained(model_dir+'/condenser/co-condenser-marco/', add_pooling_layer = False, output_hidden_states=False)
        self.pool_type = 'mean'
        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        elif loss_function == "mse":
            self.loss_fct = nn.MSELoss()
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct =  nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        self.num_labels = config.num_labels

        self.init_weights()

    def pooling(self, token_embeddings,input):
        output_vectors = []
        #attention_mask
        attention_mask = input['attention_mask']
        #[B,L]------>[B,L,1]------>[B,L,768]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        t = token_embeddings * input_mask_expanded
        #[B,768]
        sum_embeddings = torch.sum(t, 1)
 
        # [B,768]
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        output_vectors.append(sum_embeddings / sum_mask)
 
        output_vector = torch.cat(output_vectors, 1)
        return  output_vector

    def forward(
            self,
            query=None,
            attention_mask_query=None,
            token_type_ids_query=None,
            inputs_embeds_query=None,
            pos=None,
            attention_mask_pos=None,
            token_type_ids_pos=None,
            inputs_embeds_pos=None,
            neg=None,
            attention_mask_neg=None,
            token_type_ids_neg=None,
            inputs_embeds_neg=None,
            labels=None
    ):
        if inputs_embeds_pos is not None and inputs_embeds_neg is not None:
            query_emb = inputs_embeds_query
            pos_emb = inputs_embeds_pos
            neg_emb = inputs_embeds_neg
        elif query['input_ids'] is not None and pos['input_ids'] is not None and neg['input_ids'] is not None:
            query['input_ids'] = query['input_ids'].to('cuda')
            if 'attention_mask' in query:
                query['attention_mask'] = query['attention_mask'].to('cuda')
            if 'token_type_ids' in query:
                query['token_type_ids'] = query['token_type_ids'].to('cuda')
            pos['input_ids'] = pos['input_ids'].to('cuda')
            if 'attention_mask' in pos:
                pos['attention_mask'] = pos['attention_mask'].to('cuda')
            if 'token_type_ids' in pos:
                pos['token_type_ids'] = pos['token_type_ids'].to('cuda')
            neg['input_ids'] = neg['input_ids'].to('cuda')
            if 'attention_mask' in neg: 
                neg['attention_mask'] = neg['attention_mask'].to('cuda')
            if 'token_type_ids' in neg:
                neg['token_type_ids'] = neg['token_type_ids'].to('cuda')
            # print("@#",pos)
            query_emb = self.condenser(**query)[0][:,0,:].squeeze()
            pos_emb = self.condenser(**pos)[0][:,0,:].squeeze()
            neg_emb = self.condenser(**neg)[0][:,0,:].squeeze()
        if len(query_emb.size()) == 1:
            query_emb = query_emb.unsqueeze(0)
            pos_emb = pos_emb.unsqueeze(0)
            neg_emb = neg_emb.unsqueeze(0)
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=1)
        neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=1)
        similarity_pos = compute_similarity(query_emb, pos_emb)
        similarity_neg = compute_similarity(query_emb, neg_emb)

        similarity_pos_norm = norm_sim(similarity_pos, query_emb, pos_emb)
        similarity_neg_norm = norm_sim(similarity_neg, query_emb, neg_emb)
        similarity_diff = similarity_pos_norm - similarity_neg_norm
        # similarity_diff = torch.cat((similarity_neg_norm.unsqueeze(-1), similarity_pos_norm.unsqueeze(-1)), dim=1)
        loss = None
        similarity_pos = torch.tensor(similarity_pos)
        if labels is not None:
            # loss = self.loss_fct(similarity_diff.view(-1, self.num_labels), labels.view(-1))
            # loss = self.loss_fct(similarity_diff.view(-1, self.num_labels), labels.view(-1, self.num_labels))#cross
            loss = self.loss_fct(similarity_diff, labels)
            # loss = self.loss_fct(similarity_pos.to(self.device), labels)
        output = (similarity_pos.to(self.device), similarity_diff.to(self.device))
        return ((loss,) + output) if loss is not None else output
    
    def get_input_embeddings(self):
        return self.condenser.get_input_embeddings()

class CondenserForPairwiseModel_msmarco(BertPreTrainedModel):
    def __init__(self, config, loss_function="mse", smoothing=0.1) -> None:
        super().__init__(config)
        self.condenser = AutoModel.from_pretrained(model_dir+"/condenser/msmarco-bert-co-condensor")
        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        elif loss_function == "mse":
            self.loss_fct = nn.MSELoss()
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct =  nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        self.num_labels = config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir+"/condenser/msmarco-bert-co-condensor")

        self.init_weights()
    
    def cls_pooling(self, model_output):
        return model_output.last_hidden_state[:,0]
    
    def encode(self, texts):
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        encoded_input.to('cuda')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.condenser(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = self.cls_pooling(model_output)

        return embeddings
    
    def encode_(self, text_tokenize):
        # Tokenize sentences
        # encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        text_tokenize.to('cuda')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.condenser(**text_tokenize, return_dict=True)

        # Perform pooling
        embeddings = self.cls_pooling(model_output)

        return embeddings

    def forward(
            self,
            query=None,
            attention_mask_query=None,
            token_type_ids_query=None,
            inputs_embeds_query=None,
            pos=None,
            attention_mask_pos=None,
            token_type_ids_pos=None,
            inputs_embeds_pos=None,
            neg=None,
            attention_mask_neg=None,
            token_type_ids_neg=None,
            inputs_embeds_neg=None,
            labels=None
    ):
        if inputs_embeds_pos is not None and inputs_embeds_neg is not None:
            query_emb = inputs_embeds_query
            pos_emb = inputs_embeds_pos
            neg_emb = inputs_embeds_neg
        elif query['input_ids'] is not None and pos['input_ids'] is not None and neg['input_ids'] is not None:
            query['input_ids'] = query['input_ids'].to('cuda')
            if 'attention_mask' in query:
                query['attention_mask'] = query['attention_mask'].to('cuda')
            if 'token_type_ids' in query:
                query['token_type_ids'] = query['token_type_ids'].to('cuda')
            pos['input_ids'] = pos['input_ids'].to('cuda')
            if 'attention_mask' in pos:
                pos['attention_mask'] = pos['attention_mask'].to('cuda')
            if 'token_type_ids' in pos:
                pos['token_type_ids'] = pos['token_type_ids'].to('cuda')
            neg['input_ids'] = neg['input_ids'].to('cuda')
            if 'attention_mask' in neg: 
                neg['attention_mask'] = neg['attention_mask'].to('cuda')
            if 'token_type_ids' in neg:
                neg['token_type_ids'] = neg['token_type_ids'].to('cuda')
            # Compute token embeddings
            with torch.no_grad():
                model_output_q = self.condenser(**query, return_dict=True)
                model_output_pos = self.condenser(**pos, return_dict=True)
                model_output_neg = self.condenser(**neg, return_dict=True)
            query_emb = self.cls_pooling(model_output_q)
            pos_emb = self.cls_pooling(model_output_pos)
            neg_emb = self.cls_pooling(model_output_neg)
        # if len(query_emb.size()) == 1:
        #     query_emb = query_emb.unsqueeze(0)
        #     pos_emb = pos_emb.unsqueeze(0)
        #     neg_emb = neg_emb.unsqueeze(0)
        # query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        # pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=1)
        # neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=1)
        similarity_pos = torch.mm(query_emb, pos_emb.transpose(0, 1)).cpu()
        similarity_pos = np.diag(similarity_pos, k=0).tolist()
        similarity_neg = torch.mm(query_emb, neg_emb.transpose(0, 1)).cpu()
        similarity_neg = np.diag(similarity_neg, k=0).tolist()

        # similarity_pos_norm = norm_sim(similarity_pos, query_emb, pos_emb)
        # similarity_neg_norm = norm_sim(similarity_neg, query_emb, neg_emb)
        similarity_diff = np.array(similarity_pos) - np.array(similarity_neg)
        # similarity_diff = torch.cat((similarity_neg_norm.unsqueeze(-1), similarity_pos_norm.unsqueeze(-1)), dim=1)
        loss = None
        similarity_pos = torch.tensor(similarity_pos)
        similarity_diff = torch.tensor(similarity_diff)
        if labels is not None:
            # loss = self.loss_fct(similarity_diff.view(-1, self.num_labels), labels.view(-1))
            # loss = self.loss_fct(similarity_diff.view(-1, self.num_labels), labels.view(-1, self.num_labels))#cross
            loss = self.loss_fct(similarity_diff, labels)
            # loss = self.loss_fct(similarity_pos.to(self.device), labels)
        output = (similarity_pos.to(self.device), similarity_diff.to(self.device))
        return ((loss,) + output) if loss is not None else output
    
    def get_input_embeddings(self):
        return self.condenser.get_input_embeddings()