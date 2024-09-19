"""
Module contains models necessary to learn food insecurity prediction metric 
"""

import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class BertGist(nn.Module):
    """
    BertGist Model addopted from pretrained BERT on huggingface library built on pytorch classes
    """

    def __init__(self):
        """
        Constructor for the deep attention model, layers are adopted from pytorch and weights pretrained on BERT LM
        :param num_labels: no of corpus target classes defines models last layer architecture/ 1 means regression
        """
        super().__init__()
        #self.dv = device
        self.bert = AutoModelForSequenceClassification.from_pretrained("flaubert/flaubert_base_cased", num_labels=3)

    def forward(self, batch):
        """
        Main network architecture as defined in pytorch library.
        :param batch is the object containing tokenized inputs and attention positions
        :return: logits which is the food insecurity index prediction 
        """
        #print("HAPA: ",batch)
            
        output = self.bert(**batch)
        
        return output
