from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import copy
from RelationExtraction.config import Config
import torch.nn.functional as F
import numpy as np
import json
import torch

def token_transform(path=Config.path_bert + 'vocab.txt'):
    word2id = {}
    id2word = {}
    
    with (open(path, 'r', encoding='UTF8') as f):
        i = 0
        lines = f.readlines()
        for line in lines:
            word2id[line.strip()] = i
            id2word[i] = line.strip()
            i += 1
    
    return word2id, id2word

#list that is like [[text, subject, object, predicate], ...]
def extract(path) -> list:
    l = []

    with open(path, "r", encoding='UTF8') as f:
        lines = f.readlines()
        
        for line in lines:
            data = json.loads(line)
            text = data["text"]
            tmp = []
            spo_list = data["spo_list"]
            for spo in spo_list:
                tmp.append(text)
                subject = spo["subject"]
                tmp.append(subject)
                object_value = spo["object"]["@value"]
                tmp.append(object_value)
                predicate = spo["predicate"]
                tmp.append(predicate)   
                l.append(copy.deepcopy(tmp)) 
                tmp.clear()
    return l     

def establish_label_dict(data_path : str):
    label2id = {}
    items = extract(data_path)
    for item in items:
        if (item[3] not in label2id):
            label2id[item[3]] = len(label2id)
    
    return label2id

def list_pad(lst, target_length, padding_value):
    if len(lst) >= target_length:
        return lst[:target_length]
    else:
        padding_length = target_length - len(lst)
        padding = [padding_value] * padding_length
        return lst + padding


class REDataset(Dataset):
    def __init__(self, data_path : str, label2id : dict):
        items = extract(data_path)
        self.sentences = []
        self.sentences_level_label = []
        self.masks = []
        self.segments_id = []

        tokenizer = BertTokenizer.from_pretrained(Config.path_bert)
        special_tokens_dict = {'additional_special_tokens':['[unused1]', '[unused2]']}
        tokenizer.add_special_tokens(special_tokens_dict)

        for item in items:
            text = item[0]
            sub = item[1]
            obj = item[2]
            predicate = item[3]
            # if len(text) + 16 > Config.max_seq_len:
            #     text = text[:Config.max_seq_len - 16]
            text = text.replace(sub, '[SEP][unused1]').replace(obj, '[SEP][unused2]')

            tokens = tokenizer.tokenize(text)
            tmp = tokenizer.encode_plus(tokens, add_special_tokens=True, max_length=Config.max_seq_len, padding='max_length')            

            input_ids = tmp['input_ids']
            attention_mask = tmp['attention_mask']
            # print(f'len of text = {len(text)}, len of input_id = {len(input_ids)}, len of mask = {len(attention_mask)}')
            self.sentences.append(input_ids)
            self.sentences_level_label.append([label2id[predicate]])
            self.masks.append(attention_mask)
            self.segments_id.append(list_pad([1], Config.max_seq_len, 0))
            pass
        
        pass

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        return self.sentences[index], self.sentences_level_label[index], self.masks[index], self.segments_id[index]