from NamedEntityRecognition.model import NERModel
import torch
from NamedEntityRecognition.data_loader import establish_label_dict, list_pad
from NamedEntityRecognition.config import Config
import numpy as np
from transformers import BertTokenizer


label2id = establish_label_dict(Config.path_train)
id2label = {y: x for x, y in label2id.items()}
tokenizer = BertTokenizer.from_pretrained(Config.path_bert)

def extract_entities_from_tokens_and_tags(tokens, tags):
    entities = []
    entity = None

    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            entity_type = tag.split('-')[1]
            if entity:
                entities.append((entity, entity_type))
            entity = token
        elif tag.startswith('I-'):
            if entity:
                entity += token
        else:
            if entity:
                entities.append((entity, entity_type))
                entity = None

    if entity:
        entities.append((entity, entity_type))

    return entities

def extract_entities(text : str):

    data = tokenizer.encode(text, max_length=Config.max_seq_len, padding='max_length', add_special_tokens=False)
    
    true_len = len(text)
    mask = list_pad([1] * true_len, Config.max_seq_len, 0)
    segment_id = list_pad([1], Config.max_seq_len, 0)

    data = torch.Tensor(data)
    mask = torch.Tensor(mask)
    segment_id = torch.Tensor(segment_id)

    data = torch.unsqueeze(data, dim=0)
    data = data.int()
    mask = torch.unsqueeze(mask, dim=0)
    mask = mask.int()
    segment_id = torch.unsqueeze(segment_id, dim=0)
    segment_id = segment_id.int()

    model = NERModel()
    state_dict = torch.load('NamedEntityRecognition/model/model.pt')
    model.load_state_dict(state_dict=state_dict)
    # model = torch.load('NamedEntityRecognition/model/model.pth')

    output = model(data, mask, segment_id)
    _, predict = torch.max(output, dim=2)
    predict = torch.squeeze(predict, dim=0)
    predict = predict[0:true_len]

    token_list = []
    predict_label = []

    for word, i in zip(text, predict):
        token_list.append(word)
        predict_label.append(id2label[i.item()])

    return extract_entities_from_tokens_and_tags(token_list, predict_label)




    

    
