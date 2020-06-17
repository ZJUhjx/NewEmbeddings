import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


###########################################################################################################
# Config parameters
## 使用时根据选用不同的embedding matrix设置不同的embedding_source和embedding_size
class Config(object):
    sequence_length = 200
    batch_size = 128
    vocab_size = 58080

    data_source = "data/train_data.csv"
    embedding_source = 'embeddings/attn_embedding.npy'
    word2idx_source = 'data/word2idx.json'
    label2idx_source = 'data/label2idx.json'

    num_classes = 2  # 二分类
    rate = 0.8  # 训练集的比例

    embedding_size = 256
    num_filters = 128
    filter_sizes = [2, 3, 4, 5]
    dropout = 0.5

    epoches = 2
    evaluate_every = 100
    checkpoint_every = 100
    learning_rate = 1e-3


# 实例化配置参数对象
config = Config()


###########################################################################################################
# Define IMDBDataset
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, df, config):
        self.df = df
        self.labeled = 'sentiment' in df
        self.data_source = config.data_source

        self.sequence_length = config.sequence_length

        self.word2idx = self.load_dict(config.word2idx_source)
        self.label2idx = self.load_dict(config.label2idx_source)

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]

        reviews = self.get_input_data(row)
        data['reviews'] = reviews

        if self.labeled:
            labels = self.get_target_idx(row)
            data['labels'] = labels

        return data

    def __len__(self):
        return len(self.df)

    def get_input_data(self, row):
        # 读取review
        review = row.review.lower().strip().split()

        # 将句子数值化
        reviewId = [self.word2idx.get(word, self.word2idx['UNK']) for word in review]
        if len(reviewId) >= self.sequence_length:
            reviewId = reviewId[:self.sequence_length]
        else:
            reviewId += [self.word2idx["PAD"]] * (self.sequence_length - len(reviewId))

        return torch.tensor(reviewId)

    def get_target_idx(self, row):
        label = row.sentiment.lower().strip()
        return self.label2idx[label]

    def load_dict(self, path):
        with open(path, "r", encoding="utf-8") as f:
            item2idx = json.load(f)
        return item2idx


def get_train_val_loaders(df, train_idx, val_idx, batch_size=128):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        IMDBDataset(train_df, config),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        IMDBDataset(val_df, config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    return dataloaders_dict


def get_test_loader(df, batch_size=128):
    loader = torch.utils.data.DataLoader(
        IMDBDataset(df, config),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)
    return loader


###########################################################################################################
# Define textCNN model
class TextCNN(nn.Module):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        if config.embedding_source:
            print("Loading pretrained embedding...")
            embedding_matrix = np.load(config.embedding_source)
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(k, config.embedding_size))
             for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        # batch_size x number_filters x seq_len - kernel_size + 1
        x = F.relu_(conv(x)).squeeze(3)
        # batch_size x number_filters
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # b x 1 x s x embed
        embed = self.embedding(x).unsqueeze(1)
        out = [self.conv_and_pool(embed, conv) for conv in self.convs]
        out = torch.cat(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


###########################################################################################################
# Define loss function and evaluate function
def loss_fn(y_pred, y_true):
    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(y_pred, y_true)
    return loss


def metrics_fn(y_pred, y_true):
    acc = accuracy_score(y_pred, y_true)
    precision = precision_score(y_pred, y_true)
    recall = recall_score(y_pred, y_true)

    return acc, precision, recall


###########################################################################################################
# Define training process
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    model.cuda()

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_acc, epoch_precision, epoch_recall = 0.0, 0.0, 0.0

            for data in (dataloaders_dict[phase]):
                reviews = data['reviews'].cuda()
                labels = data['labels'].cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    logits = model(reviews)

                    loss = criterion(logits, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * len(labels)

                    y_true = labels.cpu().detach().numpy()
                    y_pred = np.argmax(torch.softmax(logits, dim=-1).cpu().detach().numpy(), axis=1)

                    acc, precision, recall = metrics_fn(y_pred, y_true)
                    epoch_acc += acc * config.batch_size
                    epoch_precision += precision * config.batch_size
                    epoch_recall += recall * config.batch_size

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_acc / len(dataloaders_dict[phase].dataset)
            epoch_precision = epoch_precision / len(dataloaders_dict[phase].dataset)
            epoch_recall = epoch_recall / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} | Loss: {:.4f} | Acc: {:.4f} | Precision: {:.4f} | Recall: {:.4f}'.format(
                epoch + 1, num_epochs, phase, epoch_loss, epoch_acc, epoch_precision, epoch_recall))

    torch.save(model.state_dict(), filename)


############################################################################################################
# Main function
num_epochs = config.epoches
batch_size = config.batch_size
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=94)

train_df = pd.read_csv('data/train_data.csv')
train_df['review'] = train_df['review'].astype(str)
train_df['sentiment'] = train_df['sentiment'].astype(str)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df.sentiment), start=1):
    print(f'Fold: {fold}')

    model = TextCNN(config)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    criterion = loss_fn
    dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

    train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        num_epochs,
        f'model/TextCNN_fold{fold}.pth')

test_df = pd.read_csv('data/test_data.csv')
test_df['review'] = test_df['review'].astype(str)
test_df['sentiment'] = test_df['sentiment'].astype(str)
test_loader = get_test_loader(test_df)
predictions = []
all_labels = []
models = []
for fold in range(skf.n_splits):
    model = TextCNN(config)
    model.cuda()
    model.load_state_dict(torch.load(f'model/TextCNN_fold{fold + 1}.pth'))
    model.eval()
    models.append(model)

for data in test_loader:
    reviews = data['reviews'].cuda()
    labels = data['labels'].cuda()
    y_true = labels.cpu().detach().numpy()
    all_labels += list(y_true)

    logits = []
    for model in models:
        with torch.no_grad():
            output = model(reviews)
            logits.append(torch.softmax(output, dim=1).cpu().detach().numpy())

    logits = np.mean(logits, axis=0)
    y_pred = np.argmax(logits, axis=1)
    predictions += list(y_pred)

acc, precision, recall = metrics_fn(predictions, all_labels)
print('Test Acc: {:.4f} | Precision: {:.4f} | Recall: {:.4f}'.format(acc, precision, recall))
