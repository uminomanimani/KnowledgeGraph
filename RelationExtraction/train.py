from RelationExtraction.model import REModel
from RelationExtraction.config import Config
import torch
import torch.nn.functional as F
from RelationExtraction.data_loader import REDataset, establish_label_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_re():
    label2id = establish_label_dict(Config.path_train)

    train_dataset = REDataset(Config.path_train, label2id=label2id)
    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    
    test_dataset = REDataset(Config.path_test, label2id=label2id)
    test_loader = DataLoader(
        test_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)

    model = REModel(Config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    for epoch in range(Config.epoch):
        l = len(train_loader)
        with tqdm(total=l, desc=f'Epoch {epoch} train') as pbar_train:
            total_loss = 0
            for data, label, mask, segment_id in train_loader:
                pbar_train.update(1)

                data = torch.stack(data, dim=1)
                data = data.to(device=device)
                label = torch.stack(label, dim=1)
                label = label.to(device=device)
                mask = torch.stack(mask, dim=1)
                mask = mask.to(device=device)
                segment_id = torch.stack(segment_id, dim=1)
                segment_id = segment_id.to(device=device)

                optimizer.zero_grad()
                # print(data.shape)
                # print(label.shape)
                # print(mask.shape)
                # print(segment_id.shape)
                model = model.to(device=device)
                model.train()
                output = model(data, mask, segment_id)

                # print(output.shape)
                # print(predict.shape)

                output = output.reshape(-1, Config.num_relations)
                label = label.reshape(-1)
                loss = loss_fn(output, label)

                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                # print(output.shape)
        
        model.eval()
        with (torch.no_grad()):
            # model = model.to('cpu')

            correct = 0.0
            total = 0.0

            ll = len(test_loader)
            with tqdm(total=ll, desc=f'Epoch {epoch}  test') as pbar_test:
                for batch in test_loader:
                    pbar_test.update(1)
                    data, label, mask, segment_id = batch
                    data = torch.stack(data, dim=1)
                    data = data.to(device=device)
                    label = torch.stack(label, dim=1)
                    label = label.to(device=device)
                    mask = torch.stack(mask, dim=1)
                    mask = mask.to(device=device)
                    segment_id = torch.stack(segment_id, dim=1)
                    segment_id = segment_id.to(device=device)

                    output = model(data, mask, segment_id)
                    _, predict = torch.max(output, dim=1, keepdim=True)

                    predict = predict.long()
                    label = label.long()

                    x = torch.eq(predict, label)

                    correct += torch.sum((torch.eq(predict, label))).item()

                    total += predict.shape[0]

            
            print(f'Epoch={epoch}, train loss={total_loss}, accuracy={correct/total}')

    model = model.to('cpu')
    torch.save(model, f"./model/model.pth")