import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from matplotlib import pyplot as plt
from tool import pplt,get_kfold_data
import time


def download(name, cache_dir=os.path.join('..', 'data')):
    assert name in DATA_HUB, f"{name} doesn't exit in {DATA_HUB}"
    url, shal_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        return fname
    print(f'正在从{url}下载{fname}')
    r = requests.get(url)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


def download_extract(name, folder=None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir


def down_all():
    for name in DATA_HUB:
        download(name)


# data load
DATA_HUB = {}
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

# data preprocessin
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True)
train_features = torch.tensor(all_features[:train_data.shape[0]].values, dtype=torch.float32,device='cuda')
test_features = torch.tensor(all_features[train_data.shape[0]:].values, dtype=torch.float32,device='cuda')
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32,device='cuda')

# net
in_features = train_features.shape[1]
net = nn.Sequential(nn.Linear(in_features, in_features//2),
                    nn.ReLU(),
                    nn.Linear(in_features//2,1))
net = net.to(torch.device('cuda'))
# train
loss = nn.MSELoss()


def log_rmse(net, features, labels):  # 为了在取对数时进⼀步稳定该值，将⼩于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = DataLoader(TensorDataset(train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append((log_rmse(net, test_features, test_labels)))
    return train_ls, test_ls


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_kfold_data(k, i, X_train, y_train)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        #pplt(list(range(1,num_epochs+1)),[train_ls,valid_ls],legend=['train_ls','valid_ls'],xlabel ='epoch',ylabel='loss',title='loss-change')
        print(f'折{i + 1},训练log_rmse{float(train_ls[-1])}',
              f'验证log_rmse{float(valid_ls[-1])}')
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将⽹络应⽤于测试集。
    preds = net(test_features).detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

a = time.time()
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.5, 0.2, 64
k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
b = time.time()
print(b-a)
#train_and_pred(train_features, test_features, train_labels, test_data,
#num_epochs, lr, weight_decay, batch_size)

