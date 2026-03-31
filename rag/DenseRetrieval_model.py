import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)
import torch
import numpy as np
from transformers import BertPreTrainedModel, BertModel, AutoModel, AutoTokenizer, AutoModelForPreTraining, AutoModelForSequenceClassification, AutoModelForPreTraining
from sentence_transformers import SentenceTransformer
# from bert_ranker.losses import label_smoothing
from torch import nn
from condenser import condenser_loss, condenser_encode, compute_similarity, norm_sim

model_hub_path = "/public/home/lab6/model_hub/"

class ContrieverForPairwiseModel(BertPreTrainedModel):#作用有：计算稠密相关性，返回loss用于梯度搜索，
    def __init__(self, config, loss_function="mse", smoothing=0.1) -> None:
        super().__init__(config)
        self.embedding_model = AutoModel.from_pretrained(model_hub_path+"contriever_msmarco")
        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        elif loss_function == "mse":
            self.loss_fct = nn.MSELoss()
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct =  nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        self.num_labels = config.num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_hub_path+"contriever_msmarco")

        self.init_weights()
    
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # Tokenize sentences
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        encoded_input.to('cuda')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input, return_dict=True)

        # Perform pooling
        embeddings = self.mean_pooling(model_output[0], encoded_input['attention_mask'])

        return embeddings
    
    def encode_(self, text_tokenize):
        # Tokenize sentences
        text_tokenize.to('cuda')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embedding_model(**text_tokenize, return_dict=True)

        # Perform pooling
        embeddings = self.cls_pooling(model_output[0], text_tokenize['attention_mask'])

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
                model_output_q = self.embedding_model(**query, return_dict=True)
                model_output_pos = self.embedding_model(**pos, return_dict=True)
                model_output_neg = self.embedding_model(**neg, return_dict=True)
            query_emb = self.mean_pooling(model_output_q[0], query['attention_mask'])
            pos_emb = self.mean_pooling(model_output_pos[0], pos['attention_mask'])
            neg_emb = self.mean_pooling(model_output_neg[0], neg['attention_mask'])

        similarity_pos = torch.mm(query_emb, pos_emb.transpose(0, 1)).cpu()
        similarity_pos = np.diag(similarity_pos, k=0).tolist()
        # print(torch.mm(query_emb, pos_emb.transpose(0, 1)))
        # print(similarity_pos)
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
        return self.embedding_model.get_input_embeddings()

class ANCEForPairwiseModel():
    def __init__(self, mode_name, loss_function="mse", smoothing=0.1) -> None:
        # super().__init__(config)
        # self.embedding_model = AutoModel.from_pretrained("/mnt/data_share/model_hub/msmarco-roberta-base-ance-firstp")
        self.embedding_model = SentenceTransformer(model_hub_path+'msmarco-roberta-base-ance-firstp')
        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        elif loss_function == "mse":
            self.loss_fct = nn.MSELoss()
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct =  nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        self.num_labels = 2
        self.tokenizer = AutoTokenizer.from_pretrained(model_hub_path+"msmarco-roberta-base-ance-firstp")


    def cls_pooling(self ,model_output, attention_mask):
        return model_output.last_hidden_state[:,0]
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # Tokenize sentences
        # encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        # encoded_input.to('cuda')

        # # Compute token embeddings
        # with torch.no_grad():
        #     model_output = self.embedding_model(**encoded_input, return_dict=True)

        # # Perform pooling
        # # print(model_output[0])
        # # print(encoded_input['attention_mask'])
        # embeddings = self.cls_pooling(model_output, encoded_input['attention_mask'])

        embeddings = self.embedding_model.encode(texts)

        return embeddings
    
    def encode_(self, text_tokenize):
        # Tokenize sentences
        # encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        text_tokenize.to('cuda')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embedding_model(**text_tokenize, return_dict=True)

        # Perform pooling
        embeddings = self.cls_pooling(model_output, encoded_input['attention_mask'])

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
                model_output_q = self.embedding_model(**query, return_dict=True)
                model_output_pos = self.embedding_model(**pos, return_dict=True)
                model_output_neg = self.embedding_model(**neg, return_dict=True)
            query_emb = self.cls_pooling(model_output_q, query['attention_mask'])
            pos_emb = self.cls_pooling(model_output_pos, pos['attention_mask'])
            neg_emb = self.cls_pooling(model_output_neg, neg['attention_mask'])
        # if len(query_emb.size()) == 1:
        #     query_emb = query_emb.unsqueeze(0)
        #     pos_emb = pos_emb.unsqueeze(0)
        #     neg_emb = neg_emb.unsqueeze(0)
        # query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        # pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=1)
        # neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=1)

        similarity_pos = torch.mm(query_emb, pos_emb.transpose(0, 1)).cpu()
        similarity_pos = np.diag(similarity_pos, k=0).tolist()
        # print(torch.mm(query_emb, pos_emb.transpose(0, 1)))
        # print(similarity_pos)
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
        return self.embedding_model.get_input_embeddings()

    def eval(self,):
        self.embedding_model.eval()
    
    def to(self, device):
        self.embedding_model.to(device)


class DPRForPairwiseModel():
    def __init__(self, model_name=model_hub_path+'dpr', loss_function="mse", smoothing=0.1, language_code="en_XX") -> None:
        self.embedding_model = SentenceTransformer(model_name)
        # Activate the language-specific adapters
        self.embedding_model[0].auto_model.set_default_language(language_code)
        
        # Initialize loss function
        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss()
        elif loss_function == "mse":
            self.loss_fct = nn.MSELoss()
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss()
        self.num_labels = 2

    def cls_pooling(self, model_output, attention_mask):
        # Take the representation from the CLS token
        return model_output.last_hidden_state[:, 0]

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # Compute embeddings with normalization
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
        return embeddings

    def forward(
            self,
            query_texts=None,
            pos_texts=None,
            neg_texts=None,
            labels=None
    ):
        device = next(self.embedding_model.parameters()).device
        # Encode the queries and passages
        query_emb = self.embedding_model.encode(query_texts, normalize_embeddings=True, convert_to_tensor=True).to(device)
        pos_emb = self.embedding_model.encode(pos_texts, normalize_embeddings=True, convert_to_tensor=True).to(device)
        neg_emb = self.embedding_model.encode(neg_texts, normalize_embeddings=True, convert_to_tensor=True).to(device)

        # Compute similarities
        similarity_pos = torch.sum(query_emb * pos_emb, dim=-1)
        similarity_neg = torch.sum(query_emb * neg_emb, dim=-1)

        # Compute similarity differences
        similarity_diff = similarity_pos - similarity_neg

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss = self.loss_fct(similarity_diff, labels)

        output = (similarity_pos, similarity_diff)
        return ((loss,) + output) if loss is not None else output

    def get_input_embeddings(self):
        return self.embedding_model[0].auto_model.get_input_embeddings()

    def eval(self):
        self.embedding_model.eval()

    def to(self, device):
        self.embedding_model.to(device)