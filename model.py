import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from config import Config

class NERModel(nn.Module):
    def __init__(self, hidden_size=Config.hidden_size):
        super(NERModel, self).__init__()
        self.bert = BertModel.from_pretrained(Config.path_bert)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(in_features=hidden_size, out_features=Config.num_entities, bias=True)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tokens, input_mask, segment_ids, hidden_size=Config.hidden_size):
        hidden_states = self.bert(input_tokens, attention_mask=input_mask, token_type_ids=segment_ids)[0]
        return self.linear(self.dropout(hidden_states))
