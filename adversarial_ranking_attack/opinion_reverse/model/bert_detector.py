import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from transformers import BertPreTrainedModel, BertModel, AutoModel, AutoTokenizer
from bert_ranker.losses import label_smoothing
from torch import nn
import torch

class Bert_for_detection(BertPreTrainedModel):
    def __init__(self, config, loss_function="cross-entropy", smoothing=0.1):
        super().__init__(config)
        # print("bert_config:",config.num_labels)
        self.num_out = 3
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_out)
        self.softmax = nn.Softmax(dim = 1)

        if loss_function == "cross-entropy":
            self.loss_fct = nn.CrossEntropyLoss(size_average=False, reduce=True, reduction=None)
        elif loss_function == "label-smoothing-cross-entropy":
            self.loss_fct = label_smoothing.LabelSmoothingCrossEntropy(smoothing)
        else:
            self.loss_fct = nn.MSELoss()

        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None
    ):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        #print("output:", output)
        pooled_output = output[1]
        dropout_out = self.dropout(pooled_output)
        logits = self.classifier(dropout_out)
        pos_logit = self.softmax(logits)
        #print("pos_logit", pos_logit)
        #pos_logit = pos_logit[:,1]
        pos_logit.to(torch.float32)
        #print("after_logot:", pos_logit)
        loss = None
        if labels is not None:
            loss = self.loss_fct(pos_logit, labels)
            loss.to(torch.float32)
        
        outputs = tuple(pos_logit)
        #print("before:", output)
        return (loss,pos_logit) if loss is not None else outputs
