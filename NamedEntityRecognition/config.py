class Config:
    batch_size = 256
    max_seq_len = 150
    learning_rate = 1e-5
    epoch = 20
    hidden_size = 1024
    weight_decay = 0
    path_bert = "NamedEntityRecognition/../Bert/"
    num_entities = 9
    path_train = 'NamedEntityRecognition/dataset/example.train'
    path_test = 'NamedEntityRecognition/dataset/example.test'