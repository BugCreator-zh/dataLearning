import torch
from torch import nn

class VGG(nn.Module):
    def __init__(self,features):
        super(VGG,self).__init__()
        self.features = features

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(4096,10)
        )

    def forward(self,img):
        features = self.features(img)
        output = self.fc(features)
        return output