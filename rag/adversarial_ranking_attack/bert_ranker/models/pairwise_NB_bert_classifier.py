"""
@Reference: https://github1s.com/Guzpenha/transformer_rankers/blob/HEAD/transformer_rankers
"""
import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

from transformers import BertPreTrainedModel, BertModel, AutoModel, AutoTokenizer, AutoModelForPreTraining, AutoModelForSequenceClassification
from bert_ranker.losses import label_smoothing
from torch import nn


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
