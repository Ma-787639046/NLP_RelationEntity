#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT RE Model Implementation
See: Soares, Livio Baldini, et al. "Matching the Blanks: Distributional Similarity for Relation Learning." 
     Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019. 

@author: Ma
@data: 2020/12/21

"""
from transformers import BertModel
import torch
import torch.nn as nn

class BertmodelRE(BertModel):
    def __init__(self, config, n_classes=19):
        # Initiate Parent Class BertModel with config
        super(BertmodelRE, self).__init__(config)
        self.config = config

        # One Classification Layer
        self.n_classes = n_classes
        self.classification_layer = nn.Linear(2 * config.hidden_size, n_classes)
    
    def forward(self, *input, **kwargs):
        e1_e2_start = kwargs.pop('e1_e2_start')
        bert_output = super(BertmodelRE, self).forward(*input, **kwargs)
        sequence_output = bert_output[0]
        blankv1v2 = sequence_output[:, e1_e2_start, :]
        buffer = []
        for i in range(blankv1v2.shape[0]): # iterate batch & collect
            v1v2 = blankv1v2[i, i, :, :]
            v1v2 = torch.cat((v1v2[0], v1v2[1]))
            buffer.append(v1v2)
        del blankv1v2
        v1v2 = torch.stack([a for a in buffer], dim=0)
        del buffer

        classification_logits = self.classification_layer(v1v2)
        return classification_logits

# model = BertmodelRE.from_pretrained("./model/bert-base-uncased", n_classes=19)
# print(model)
