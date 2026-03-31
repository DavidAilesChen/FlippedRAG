import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)
import torch
from transformers import BertPreTrainedModel, BertModel, AutoModel, AutoTokenizer, AutoModelForPreTraining, AutoModelForSequenceClassification
from torch import nn
from condenser import compute_similarity, norm_sim
import label_smoothing

class SurrogatePairwiseModel(BertPreTrainedModel):
    def __init__(self, model_path ,config, loss_function="mse", smoothing=0.1) -> None:
        super().__init__(config)
        self.condenser = AutoModel.from_pretrained(model_path, add_pooling_layer = False, output_hidden_states=False)
        self.pool_type = 'mean'
        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        elif loss_function == "mse":
            self.loss_fct = nn.MSELoss()
        # elif loss_function == "label-smoothing-cross-entropy":
            # self.loss_fct = label_smoothing.LabelSmoothingCrossEntropy(smoothing)
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
    
class MinitForPairwiseLearning(BertPreTrainedModel):
    """
    """
    def __init__(self, config, loss_function="label-smoothing-cross-entropy", smoothing=0.1):
        super().__init__(config)
        print("mini_pariwise_config:",config.num_labels)

        #There should be at least relevant and non relevant options.
        self.num_labels = config.num_labels+1
        self.miniLM = AutoModel.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)#0.1
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)#hidden_size=384

        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct = label_smoothing.LabelSmoothingCrossEntropy(smoothing)

        self.init_weights()

    def forward(
        self,
        input_ids_pos=None,
        attention_mask_pos=None,
        token_type_ids_pos=None,
        inputs_embeds_pos=None,
        input_ids_neg=None,
        attention_mask_neg=None,
        token_type_ids_neg=None,
        inputs_embeds_neg=None,
        labels=None
    ):
        #forward pass for positive instances
        outputs_pos = self.miniLM(
            input_ids=input_ids_pos,
            attention_mask=attention_mask_pos,
            token_type_ids=token_type_ids_pos,
            inputs_embeds=inputs_embeds_pos
        )
        pooled_output_pos = outputs_pos[1]
        pooled_output_pos = self.dropout(pooled_output_pos)
        logits_pos = self.classifier(pooled_output_pos)

        #forward pass for negative instances
        outputs_neg = self.miniLM(
            input_ids=input_ids_neg,
            attention_mask=attention_mask_neg,
            token_type_ids=token_type_ids_neg,
            inputs_embeds=inputs_embeds_neg
        )
        pooled_output_neg = outputs_neg[1]
        pooled_output_neg = self.dropout(pooled_output_neg)
        logits_neg = self.classifier(pooled_output_neg)

        logits_diff = logits_pos - logits_neg

        # Calculating Cross entropy loss for pairs <q,d1,d2>
        # based on "Learning to Rank using Gradient Descent" 2005 ICML
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits_diff.view(-1, self.num_labels), labels.view(-1))

        # for label, we only consider the first part
        # output = (logits_pos,) + outputs_pos[2:]
        output = (logits_pos, logits_diff)
        return ((loss,) + output) if loss is not None else output
    
    def get_input_embeddings(self):
        return self.miniLM.get_input_embeddings()
    
class BertForPairwiseLearning(BertPreTrainedModel):
    """
    BERT based model for pairwise learning. It expects both the <q, positive_doc> and the <q, negative_doc>
    for doing the forward pass. The loss is cross-entropy for the difference between positive_doc and negative_doc
    scores (labels are 1 if score positive_neg > score negative_doc otherwise 0) based on 
    "Learning to Rank using Gradient Descent" 2005 ICML.
    """
    def __init__(self, config, loss_function="label-smoothing-cross-entropy", smoothing=0.1):
        super().__init__(config)

        #There should be at least relevant and non relevant options.
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=True)
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct = label_smoothing.LabelSmoothingCrossEntropy(smoothing)

        self.init_weights()

    def forward(
        self,
        input_ids_pos=None,
        attention_mask_pos=None,
        token_type_ids_pos=None,
        inputs_embeds_pos=None,
        input_ids_neg=None,
        attention_mask_neg=None,
        token_type_ids_neg=None,
        inputs_embeds_neg=None,
        labels=None
    ):
        #forward pass for positive instances
        outputs_pos = self.bert(
            input_ids=input_ids_pos,
            attention_mask=attention_mask_pos,
            token_type_ids=token_type_ids_pos,
            inputs_embeds=inputs_embeds_pos
        )
        pooled_output_pos = outputs_pos[1]
        pooled_output_pos = self.dropout(pooled_output_pos)
        logits_pos = self.classifier(pooled_output_pos)

        #forward pass for negative instances
        outputs_neg = self.bert(
            input_ids=input_ids_neg,
            attention_mask=attention_mask_neg,
            token_type_ids=token_type_ids_neg,
            inputs_embeds=inputs_embeds_neg
        )
        pooled_output_neg = outputs_neg[1]
        pooled_output_neg = self.dropout(pooled_output_neg)
        logits_neg = self.classifier(pooled_output_neg)

        logits_diff = logits_pos - logits_neg

        # Calculating Cross entropy loss for pairs <q,d1,d2>
        # based on "Learning to Rank using Gradient Descent" 2005 ICML
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits_diff.view(-1, self.num_labels), labels.view(-1))

        # for label, we only consider the first part
        # output = (logits_pos,) + outputs_pos[2:]
        output = (logits_pos, logits_diff)
        return ((loss,) + output) if loss is not None else output
    
    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()

class MiniForPairwiseClassfy(BertPreTrainedModel):
    """
    """
    def __init__(self, config, loss_function="label-smoothing-cross-entropy", smoothing=0.1):
        super().__init__(config)

        #There should be at least relevant and non relevant options.
        self.num_labels = config.num_labels+1
        self.miniLM = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")

        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct = label_smoothing.LabelSmoothingCrossEntropy(smoothing)

        self.init_weights()

    def forward(
        self,
        input_ids_pos=None,
        attention_mask_pos=None,
        token_type_ids_pos=None,
        inputs_embeds_pos=None,
        input_ids_neg=None,
        attention_mask_neg=None,
        token_type_ids_neg=None,
        inputs_embeds_neg=None,
        labels=None
    ):
        #forward pass for positive instances
        logits_pos = self.miniLM(
            input_ids=input_ids_pos,
            attention_mask=attention_mask_pos,
            token_type_ids=token_type_ids_pos,
            inputs_embeds=inputs_embeds_pos
        )[0]

        #forward pass for negative instances
        logits_neg  = self.miniLM(
            input_ids=input_ids_neg,
            attention_mask=attention_mask_neg,
            token_type_ids=token_type_ids_neg,
            inputs_embeds=inputs_embeds_neg
        )[0]
        logits_diff = torch.cat([logits_neg, logits_pos], dim = 1)

        # Calculating Cross entropy loss for pairs <q,d1,d2>
        # based on "Learning to Rank using Gradient Descent" 2005 ICML
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits_diff.view(-1, self.num_labels), labels.view(-1))

        # for label, we only consider the first part
        # output = (logits_pos,) + outputs_pos[2:]
        output = (logits_pos, logits_diff)
        return ((loss,) + output) if loss is not None else output
    
    def get_input_embeddings(self):
        return self.miniLM.get_input_embeddings()

class NBBERTForPairwiseClassfy(BertPreTrainedModel):
    """
    """
    def __init__(self, config, loss_function="label-smoothing-cross-entropy", smoothing=0.1):
        super().__init__(config)

        #There should be at least relevant and non relevant options.
        self.num_labels = config.num_labels
        self.NBbert = AutoModelForSequenceClassification.from_pretrained(config._name_or_path)

        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct = label_smoothing.LabelSmoothingCrossEntropy(smoothing)

        self.init_weights()

    def forward(
        self,
        input_ids_pos=None,
        attention_mask_pos=None,
        token_type_ids_pos=None,
        inputs_embeds_pos=None,
        input_ids_neg=None,
        attention_mask_neg=None,
        token_type_ids_neg=None,
        inputs_embeds_neg=None,
        labels=None
    ):
        #forward pass for positive instances
        logits_pos = self.NBbert(
            input_ids=input_ids_pos,
            attention_mask=attention_mask_pos,
            token_type_ids=token_type_ids_pos,
            inputs_embeds=inputs_embeds_pos
        )[0]

        #forward pass for negative instances
        logits_neg  = self.NBbert(
            input_ids=input_ids_neg,
            attention_mask=attention_mask_neg,
            token_type_ids=token_type_ids_neg,
            inputs_embeds=inputs_embeds_neg
        )[0]
        logits_diff = logits_pos - logits_neg

        # Calculating Cross entropy loss for pairs <q,d1,d2>
        # based on "Learning to Rank using Gradient Descent" 2005 ICML
        loss = None
        if labels is not None:
            loss = self.loss_fct(logits_diff.view(-1, self.num_labels), labels.view(-1))

        # for label, we only consider the first part
        # output = (logits_pos,) + outputs_pos[2:]
        output = (logits_pos, logits_diff)
        return ((loss,) + output) if loss is not None else output
    
    def get_input_embeddings(self):
        return self.NBbert.get_input_embeddings()

