import torch
from torch import nn
from d2l import torch as d2l
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset,TensorDataset

import tool
from tool import evaluate_accuracy
# 读取数据
batch_size = 256
train_dataset = datasets.MNIST(root='..//data',train=True,transform=transforms.ToTensor(),download=True)
test_datatest = datasets.MNIST(root='..//data',train=False,transform=transforms.ToTensor(),download=True)
train_iter = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_iter = DataLoader(dataset=test_datatest,batch_size=batch_size,shuffle=True)

# 网络
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net =  nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,padding=2),nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(16*5*5,120),nn.Sigmoid(),
            nn.Linear(120,84),nn.Sigmoid(),
            nn.Linear(84,10)
        )
    def forward(self,X):
        return self.net(X)


# train
def train(net,train_iter,test_iter,num_epochs,lr,device):
    def init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weight)
    print(f'training on {device}')
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    #animator = tool.Animator()
    train_l,train_acc,test_acc=[],[],[]
    for epoch in range(num_epochs):
        metric = tool.Accumulator(3)
        net.train()
        for i,(X,y) in enumerate(train_iter):
            optimizer.zero_grad()
            X,y = X.to(device),y.to(device)
            y_pred = net(X)
            l = loss(y_pred,y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l*X.shape[0],tool.accuracy(y_pred,y),X.shape[0])
        train_l.append(metric[0]/metric[2])
        train_acc.append(metric[1]/metric[2])
        test_acc.append(evaluate_accuracy(net,test_iter))
        print(f'loss:{train_l[-1]}   train_acc:{train_acc[-1]}   test_acc:{test_acc[-1]}')
    tool.pplt(list(range(1,num_epochs+1)),[train_l,train_acc,test_acc],xlabel='epoch',
              legend=['train_loss','train_acc','test_acc'],title='number')


net = LeNet()
num_epochs,lr,device = 10,0.9,'cuda'
train(net,lr=lr,train_iter=train_iter,test_iter = test_iter,num_epochs=num_epochs,device=device)




