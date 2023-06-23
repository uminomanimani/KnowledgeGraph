from model import NERModel
from config import Config
import torch
import torch.nn.functional as F
from data_loader import MyDataset, establish_label_dict
from torch.utils.data import DataLoader
from tqdm import tqdm


def accuracy(predictions, labels):
    correct_predictions = (predictions == labels)
    # 将布尔张量转换为整数张量
    correct_predictions_int = correct_predictions.int()
    # 沿 seq 维度求和，得到每个样本中预测正确的元素数量
    correct_counts = torch.sum(correct_predictions_int, dim=1)
    # 计算每个样本的准确率
    accuracy_per_sample = correct_counts.float() / labels.size(1)
    # 计算整个批次的平均准确率
    average_accuracy = torch.mean(accuracy_per_sample)

    return average_accuracy


if __name__ == '__main__':
    label2id = establish_label_dict('./dataset/train.txt')
    train_dataset = MyDataset('./dataset/train.txt', label2id=label2id)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.batch_size, shuffle=True)
    model = NERModel(Config.hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for epoch in range(Config.epoch):
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}') as pbar_epoch:
            total_loss = 0
            for batch in train_loader:
                pbar_epoch.update(1)
                data, label, mask, segment_id = batch

                data = torch.stack(data, dim=0).T
                data = data.to(device=device)
                label = torch.stack(label, dim=0).T
                label = label.to(device=device)
                mask = torch.stack(mask, dim=0).T
                mask = mask.to(device=device)
                segment_id = torch.stack(segment_id, dim=0).T
                segment_id = segment_id.to(device=device)

                optimizer.zero_grad()
                # print(data.shape)
                # print(label.shape)
                # print(mask.shape)
                # print(segment_id.shape)
                model = model.to(device=device)
                model.train()
                output = model(data, mask, segment_id)

                _, predict = torch.max(output, dim=2)
                # print(predict.shape)

                output = output.reshape(-1, Config.num_entities)
                label = label.reshape(-1)
                loss = loss_fn(output, label)

                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                # print(output.shape)

        model.eval()
        with (torch.no_grad()):
            test_dataset = MyDataset('./dataset/test.txt', label2id=label2id)
            test_loader = DataLoader(
                test_dataset, batch_size=Config.batch_size, shuffle=True)
            model.to('cpu')

            correct = 0.0
            total = 0.0

            for batch in test_loader:
                data, label, mask, segment_id = batch
                data = torch.stack(data, dim=0).T
                label = torch.stack(label, dim=0).T
                mask = torch.stack(mask, dim=0).T
                segment_id = torch.stack(segment_id, dim=0).T

                output = model(data, mask, segment_id)
                _, predict = torch.max(output, dim=2)

                non_padding_indices = torch.nonzero(mask, as_tuple=False)
                non_padding_output = output[non_padding_indices[:, 0], non_padding_indices[:, 1], :]
                non_padding_label = label[non_padding_indices[:, 0], non_padding_indices[:, 1]]
                real_length_sum = (mask == 1).sum().item()
                correct_label_sum = (non_padding_output.argmax(dim=1) == non_padding_label).sum().item()

                correct += correct_label_sum
                total += real_length_sum
            
            print(f'Epoch={epoch}, train loss={total_loss}, accuracy={correct/total}')

    model = model.to('cpu')
    torch.save(model, f"./model/model.pth")
