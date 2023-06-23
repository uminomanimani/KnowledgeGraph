from torch.utils.data import Dataset
import copy
from config import Config
import torch.nn.functional as F
import numpy as np

def token_transform(path='./Bert/vocab.txt'):
    word2id = {}
    id2word = {}
    
    with (open(path, 'r', encoding='UTF8') as f):
        i = 0
        line = f.readline()
        word2id[line.strip()] = i
        id2word[i] = line.strip()
        while (line):
            i += 1
            line = f.readline()
            word2id[line.strip()] = i
            id2word[i] = line.strip()
    
    return word2id, id2word

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
        line = f.readline()
        if '\u3000' in line:
            line = line.replace('\u3000', '[unused1]')
        label = line.strip().split(' ')[1]
        if label not in label2id:
            label2id[label] = len(label2id)
        
        while (line):
            line = f.readline()
            if '\u3000' in line:
                line = line.replace('\u3000', '[unused1]') 
            
            if len(line.strip()) != 0:
                label = line.strip().split(' ')[1]
                if label not in label2id:
                    label2id[label] = len(label2id)
    return label2id


class MyDataset(Dataset):
    def __init__(self, data_path : str, label2id : dict, vocab_path='./Bert/vocab.txt'):
        word2id, _ = token_transform(vocab_path)

        self.sentence = []
        self.sentence_label = []
        self.masks = []
        self.segment_ids = []

        # seq = []

        word_list = []
        label_list = []

        with (open(data_path, 'r', encoding='UTF8') as f):
            line = f.readline()
            if '\u3000' in line:
                    line = line.replace('\u3000', '[unused1]')
            # seq.append(line.strip()[0])
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[1]

            word_list.append(word2id[word])

            while (line):
                line = f.readline()
                if '\u3000' in line:
                    line = line.replace('\u3000', '[unused1]')  
                
                if len(line.strip()) == 0:
                    # self.sentence.append(copy.deepcopy(word_list))
                    # self.sentence_label.append(copy.deepcopy(label_list))
                    self.sentence.append(list_pad(word_list, Config.max_seq_len, word2id['[PAD]']))
                    self.sentence_label.append(list_pad(label_list, Config.max_seq_len, 0))

                    mask = [1] * len(word_list)
                    self.masks.append(list_pad(mask, Config.max_seq_len, 0))

                    segment_id = list_pad([1], Config.max_seq_len, 0)
                    self.segment_ids.append(copy.deepcopy(segment_id))


                    word_list.clear()
                    label_list.clear()
                
                else:
                    # seq.append(line.strip()[0])     
                    word = line.strip().split(' ') [0]
                    label = line.strip().split(' ') [1]
                    word_list.append(word2id[word] if word in word2id else word2id['[UNK]'])
                    label_list.append(label2id[label])
        pass

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, index):
        return self.sentence[index], self.sentence_label[index], self.masks[index], self.segment_ids[index]

if __name__ == '__main__':
    word2id, id2word = token_transform()
    print('-----')
    print(word2id)
    print(id2word)