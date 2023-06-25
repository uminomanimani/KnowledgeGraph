class Config:
    batch_size = 128
    max_seq_len = 300
    learning_rate = 1e-4
    epoch = 2
    hidden_size = 1024
    weight_decay = 0
    path_bert = "RelationExtraction/../Bert/"
    num_relations = 5
    path_train = 'RelationExtraction/dataset/train.json'
    path_test = 'RelationExtraction/dataset/test_lite.json'