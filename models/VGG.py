import torch
from torch import nn
import tool
import d2l

class VGG(nn.Module):
    def __init__(self,vgg):
        super(VGG,self).__init__()
        self.conv_layer = {
            'vgg_11': [64, 'p', 128, 'p', 256, 256, 'p', 512, 512, 'p', 512, 512, 'p'],
            'vgg_13': [64, 64, 'p', 128, 128, 'p', 256, 256, 'p', 512, 512, 'p', 512, 512, 'p'],
            'vgg_16': [64, 64, 'p', 128, 128, 'p', 256, 256, 256, 'p', 512, 512, 512, 'p', 512, 512, 512, 'p'],
            'vgg_19': [64, 64, 'p', 128, 128, 'p', 256, 256, 256, 256, 'p', 512, 512, 512, 512, 'p', 512, 512, 512, 512, 'p']
        }

        self.features = self.make_layer(vgg)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(4096,10)
        )


    def make_layer(self,vgg):
        layers = []
        in_channel = 3
        for conv in self.conv_layer[vgg]:
            if conv == 'p':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                continue

            layers.append(nn.Conv2d(in_channel, conv, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(conv))
            layers.append(nn.ReLU())
            in_channel = conv
        return nn.Sequential(*layers)

    def forward(self,img):
        features = self.features(img)
        output = self.fc(features)
        return output


def train(net,train_iter, test_iter,num_epochs,lr,device):
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
        test_acc.append(tool.evaluate_accuracy(net,test_iter))
        print(f'loss:{train_l[-1]}   train_acc:{train_acc[-1]}   test_acc:{test_acc[-1]}')
    tool.pplt(list(range(1,num_epochs+1)),[train_l,train_acc,test_acc],xlabel='epoch',
              legend=['train_loss','train_acc','test_acc'],title='number')


train_iter,test_iter = d2l.load_data_fashion_mnist(128,resize=224)
lr,num_epochs,device = 0.01,10,'cuda'
net = VGG('vgg_11')
train(net,train_iter,test_iter,num_epochs,lr,device)