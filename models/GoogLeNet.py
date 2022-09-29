import torch
from torch import nn
from torch.nn import functional as F
import d2l
import tool


class Inception(nn.Module):
    def __init__(self,in_channels,c1,c2,c3,c4,**kwargs):
        super(Inception, self).__init__()
        self.p1 = nn.Conv2d(in_channels,c1,kernel_size=1)
        self.p2_1 = nn.Conv2d(in_channels,c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self,x):
        p1 = F.relu(self.p1(x))
        p2 = F.relu((self.p2_2(F.relu(self.p2_1(x)))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1,p2,p3,p4),dim = 1)




class GoogLeNet(nn.Module):
        def __init__(self):
            super(GoogLeNet, self).__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                Inception(192, 64, (96, 128), (16, 32), 32),
                Inception(256, 128, (128, 192), (32, 96), 64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                Inception(480, 192, (96, 208), (16, 48), 64),
                Inception(512, 160, (112, 224), (24, 64), 64),
                Inception(512, 128, (128, 256), (24, 64), 64),
                Inception(512, 112, (144, 288), (32, 64), 64),
                Inception(528, 256, (160, 320), (32, 128), 128),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                Inception(832, 256, (160, 320), (32, 128), 128),
                Inception(832, 384, (192, 384), (48, 128), 128),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(1024,10)
            )

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
        test_acc.append(tool.evaluate_accuracy(net,test_iter,device))
        print(f'epoch:{epoch+1}   loss:{train_l[-1]:.5f}   train_acc:{train_acc[-1]:.5f}   test_acc:{test_acc[-1]:.5f}')
    tool.pplt(list(range(1,num_epochs+1)),[train_l,train_acc,test_acc],xlabel='epoch',
              legend=['train_loss','train_acc','test_acc'],title='number')

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,resize=96)
net = GoogLeNet()
num_epochs,lr,device = 20,0.02,'cuda'
train(net,lr=lr,train_iter=train_iter,test_iter = test_iter,num_epochs=num_epochs,device=device)
