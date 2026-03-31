import math
import torch
import logging
from torch import nn
import numpy as np
from transformers import (BertModel, BertPreTrainedModel, AutoModel, AutoModelForSequenceClassification)
logger = logging.getLogger(__name__)


class RankingBERT_Train(BertPreTrainedModel):
    def __init__(self, config):
        super(RankingBERT_Train, self).__init__(config)
        self.bert = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
        self.init_weights()

        #self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        #self.out = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids,
                labels=None):

        attention_mask = (input_ids != 0)

        bert_pooler_output = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            )
        output = bert_pooler_output.logits[:,1].unsqueeze(-1)
        #output = self.out(self.dropout(bert_pooler_output))
        # shape = [B, 1]

        if labels is not None:

            loss_fct = nn.MarginRankingLoss(margin=1.0, reduction='mean')

            y_pos, y_neg = [], []
            for batch_index in range(len(labels)):
                label = labels[batch_index]
                if label > 0:
                    y_pos.append(output[batch_index])
                else:
                    y_neg.append(output[batch_index])
            y_pos = torch.cat(y_pos, dim=-1)
            y_neg = torch.cat(y_neg, dim=-1)
            y_true = torch.ones_like(y_pos)
            assert len(y_pos) == len(y_neg)

            loss = loss_fct(y_pos, y_neg, y_true)
            output = loss, *output
        return output

class RankingBERT_Pairwise(BertPreTrainedModel):
    """
    """
    def __init__(self, config, loss_function="label-smoothing-cross-entropy", smoothing=0.1):
        super().__init__(config)
        self.NBbert = AutoModel.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
        self.num_labels = config.num_labels
        #self.NBbert = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)#0.1
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
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
        outputs_pos = self.NBbert(
            input_ids=input_ids_pos,
            attention_mask=attention_mask_pos,
            token_type_ids=token_type_ids_pos,
            inputs_embeds=inputs_embeds_pos
        )
        pooled_output_pos = outputs_pos[1]
        pooled_output_pos = self.dropout(pooled_output_pos)
        logits_pos = self.classifier(pooled_output_pos)

        #forward pass for negative instances
        outputs_neg = self.NBbert(
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

            loss_fct = nn.MarginRankingLoss(margin=1.0, reduction='mean')
            y_pos, y_neg = [], []
            for batch_index in range(len(labels)):
                label = labels[batch_index]
                if label > 0:
                    y_pos.append(logits_diff)
                else:
                    y_neg.append(logits_diff)
            y_pos = torch.cat(y_pos, dim=-1)
            y_neg = torch.cat(y_neg, dim=-1)
            y_true = torch.ones_like(y_pos)
            assert len(y_pos) == len(y_neg)

            loss = loss_fct(y_pos, y_neg, y_true)

        # for label, we only consider the first part
        # output = (logits_pos,) + outputs_pos[2:]
        output = logits_pos[:,1].unsqueeze(-1)
        #output = logits_pos
        return (loss, *output) if loss is not None else output
    
    def get_input_embeddings(self):
        return self.NBbert.get_input_embeddings()

class RankBertForPairwise(BertPreTrainedModel):
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
        """
        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=True)
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct = label_smoothing.LabelSmoothingCrossEntropy(smoothing)
        """
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
            loss_fct = nn.MarginRankingLoss(margin=1.0, reduction='mean')

            y_pos, y_neg = [], []
            for batch_index in range(len(labels)):
                label = labels[batch_index]
                if label > 0:
                    y_pos.append(logits_diff[batch_index][1])
                else:
                    y_neg.append(logits_diff[batch_index][1])
            y_pos = torch.cat(y_pos, dim=-1)
            y_neg = torch.cat(y_neg, dim=-1)
            y_true = torch.ones_like(y_pos)
            assert len(y_pos) == len(y_neg)

            loss = loss_fct(y_pos, y_neg, y_true)

        # for label, we only consider the first part
        # output = (logits_pos,) + outputs_pos[2:]
        output = logits_pos[:,1].unsqueeze(-1)
        return (loss, *output) if loss is not None else output