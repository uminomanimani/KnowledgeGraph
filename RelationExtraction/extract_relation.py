import torch
from transformers import BertTokenizer
from RelationExtraction.model import REModel
from RelationExtraction.config import Config
from RelationExtraction.data_loader import list_pad, establish_label_dict

def extract_relation(text : str, entity1 : str, entity2 : str) -> str:
    assert(entity1 in text)
    assert(entity2 in text)

    tokenizer = BertTokenizer.from_pretrained(Config.path_bert)

    text = text.replace(entity1, '[SEP][unused1]').replace(entity2, '[SEP][unused2]')

    tokens = tokenizer.tokenize(text)
    tmp = tokenizer.encode_plus(tokens, add_special_tokens=True, max_length=Config.max_seq_len, padding='max_length')            

    input_ids = tmp['input_ids']
    input_ids = torch.Tensor(input_ids).int()
    input_ids = torch.unsqueeze(input_ids, dim=0)
    attention_mask = tmp['attention_mask']
    attention_mask = torch.Tensor(attention_mask).int()
    attention_mask = torch.unsqueeze(attention_mask, dim=0)
    segment_id = list_pad([1], Config.max_seq_len, 0)
    segment_id = torch.Tensor(segment_id).int()
    segment_id = torch.unsqueeze(segment_id, dim=0)

    model = REModel()
    state_dict = torch.load('RelationExtraction/model/model.pt')
    model.load_state_dict(state_dict=state_dict)

    output = model(input_ids, attention_mask, segment_id)

    threshold = 0.8
    probablity = torch.softmax(output, dim=1)
    diff = probablity - threshold
    if torch.all(diff < 0):
        return None

    label2id = establish_label_dict(Config.path_train)
    id2label = {y: x for x, y in label2id.items()}

    _, predict = torch.max(output, dim=1, keepdim=True)

    predict = torch.squeeze(predict, dim=0)
    return id2label[predict.item()]

    
