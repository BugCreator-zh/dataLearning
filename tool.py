import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

def accuracy(y_hat, y):
    y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def pplt(xaxis, values, legend, xlabel=None, ylabel=None, title=None):
    """
    :param xaxis: range of axis -x
    :param values: multiple values
    :param legend: multile legend
    :param xlabel:
    :param ylabel:
    :param title:
    :return: None
    """
    for value in values:
        plt.plot(xaxis, value)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend([tag for tag in legend])
    plt.show()

# noinspection PyUnboundLocalVariable
def get_kfold_data(k,i,X,y):
    assert k>1
    fold_size = X.shape[0]//k
    train,test = None,None
    for j in range(k):
        idx = slice(j*fold_size,(j+1)*fold_size)
        if j==i:
            test,test_label = X[idx,:],y[idx]
        elif train is None:
            train,train_label = X[idx,:],y[idx]
        else:
            train,train_label = torch.cat([train,X[idx,:]],0),torch.cat([train_label,y[idx]],0)
    return train,train_label,test,test_label

def evaluate_accuracy(net,data_iter,device=None):
    if isinstance(net,nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            if isinstance(X,list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]