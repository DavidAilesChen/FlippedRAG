"""
@Reference: https://github1s.com/Guzpenha/transformer_rankers/blob/HEAD/transformer_rankers
"""
import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from transformers import BertPreTrainedModel, BertModel, AutoModel, AutoTokenizer, AutoModelForPreTraining, AutoModelForSequenceClassification
from bert_ranker.losses import label_smoothing
import torch
from torch import nn



class MinitForPairwiseLearning(BertPreTrainedModel):
    """
    """
    def __init__(self, config, loss_function="label-smoothing-cross-entropy", smoothing=0.1):
        super().__init__(config)
        print("mini_pariwise_config:",config.num_labels)

        #There should be at least relevant and non relevant options.
        self.num_labels = config.num_labels+1
        self.miniLM = AutoModel.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        #print("mini_model:", self.miniLM)
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
        #print("labels:",labels)
        pooled_output_pos = outputs_pos[1]
        pooled_output_pos = self.dropout(pooled_output_pos)
        #print(pooled_output_pos.shape)
        logits_pos = self.classifier(pooled_output_pos)
        #print("pos_logit:", logits_pos)

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
            # print(labels.shape,"@@", logits_diff.shape)
            loss = self.loss_fct(logits_diff.view(-1, self.num_labels), labels.view(-1))

        # for label, we only consider the first part
        # output = (logits_pos,) + outputs_pos[2:]
        logits_pos = torch.cat([logits_neg, logits_pos], dim = 1)
        output = (logits_pos, logits_diff)
        return ((loss,) + output) if loss is not None else output
    
    def get_input_embeddings(self):
        return self.miniLM.get_input_embeddings()
