import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from RelationExtraction.config import Config

class REModel(nn.Module):
    def __init__(self, hidden_size=Config.hidden_size, num_class=Config.num_relations) -> None:
        super(REModel, self).__init__()
        self.bert = BertModel.from_pretrained(Config.path_bert)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_class, bias=True)
    
    def forward(self, input_tokens : torch.Tensor, input_mask : torch.Tensor, segments_id : torch.Tensor):
        hidden_states = self.bert(input_tokens, attention_mask=input_mask, token_type_ids=segments_id)  #[batch_size, seq_len, hidden_size]
        return self.linear(self.dropout(hidden_states.pooler_output))  #[batch_size, hidden_size]
