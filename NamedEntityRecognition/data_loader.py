from torch.utils.data import Dataset
import copy
from NamedEntityRecognition.config import Config
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer

def list_pad(lst, target_length, padding_value):
    if len(lst) >= target_length:
        return lst[:target_length]
    else:
        padding_length = target_length - len(lst)
        padding = [padding_value] * padding_length
        return lst + padding
    
def establish_label_dict(data_path : str):
    label2id = {}
    with (open(data_path, 'r', encoding='UTF8') as f):
        lines = f.readlines()
        
        for line in lines:
            if '\u3000' in line:
                line = line.replace('\u3000', '[unused1]') 
            
            if len(line.strip()) != 0:
                label = line.strip().split(' ')[1]
                if label not in label2id:
                    label2id[label] = len(label2id)
    return label2id


class NERDataset(Dataset):
    def __init__(self, data_path : str, label2id : dict):
        tokenizer = BertTokenizer.from_pretrained(Config.path_bert)

        self.sentences = []
        self.sentences_token_level_label = []
        self.masks = []
        self.segments_id = []

        # seq = []

        word_id_list = []
        label_id_list = []

        with (open(data_path, 'r', encoding='UTF8') as f):
            lines = f.readlines()

            for line in lines:
                if '\u3000' in line:
                    line = line.replace('\u3000', '[unused1]')  
                
                if len(line.strip()) == 0:
                    # self.sentence.append(copy.deepcopy(word_id_list))
                    # self.sentences_token_level_label.append(copy.deepcopy(label_id_list))
                    self.sentences.append(list_pad(word_id_list, Config.max_seq_len, tokenizer.encode('[PAD]', add_special_tokens=False)[0]))
                    self.sentences_token_level_label.append(list_pad(label_id_list, Config.max_seq_len, label2id['O']))
                    self.masks.append(list_pad([1] * len(word_id_list), Config.max_seq_len, 0))
                    self.segments_id.append(copy.deepcopy(list_pad([1], Config.max_seq_len, 0)))

                    word_id_list.clear()
                    label_id_list.clear()
                
                else:
                    # seq.append(line.strip()[0])     
                    word = line.strip().split(' ') [0]
                    label = line.strip().split(' ') [1]
                    word_id_list.append(tokenizer.encode(word, add_special_tokens=False)[0])
                    label_id_list.append(label2id[label])
        pass

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.sentences_token_level_label[index], self.masks[index], self.segments_id[index]