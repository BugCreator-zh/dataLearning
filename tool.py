import matplotlib.pyplot as plt
import numpy as np
import torch


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