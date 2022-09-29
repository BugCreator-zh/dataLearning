import torch
from torch import nn
import tool
import d2l

class NiN(nn.Module):
    def __init__(self):
        super(NiN, self).__init__()
        self.net = nn.Sequential(
            self.nin_block(1,96,kernel_size=11,strides=4),
            nn.MaxPool2d(3,stride=2),
            self.nin_block(96,256,kernel_size=5,padding=2),
            nn.MaxPool2d(3,stride=2),
            self.nin_block(256,384,kernel_size=3,padding=1),
            nn.MaxPool2d(3,stride=2),
            nn.Dropout(0.5),
            self.nin_block(384,10,kernel_size=3,padding=1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

    def nin_block(self,in_channels,out_channels,kernel_size,strides=1,padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,strides,padding),nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=1),nn.ReLU()
        )

    def forward(self,img):
        return self.net(img)


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
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,resize=224)
net = NiN()
num_epochs,lr,device = 25,0.02,'cuda'
train(net,lr=lr,train_iter=train_iter,test_iter = test_iter,num_epochs=num_epochs,device=device)
