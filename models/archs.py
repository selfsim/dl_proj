# This class contains my model architectures
# It contains the following:
# a linear classifier with no hidden layers
# a convulutional neural network with ...
# a DNN with inception type architectures
# ???

import torch.nn as nn
import torch.nn.functional as f

class linearClassifier(nn.Module):
    def __init__(self, input_dim, classes):
        super(linearClassifier, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=classes)

    def forward(self, x):
        return self.linear(x)

class convNet(nn.Module):
    def __init__(self, classes):
        super(convNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.c1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout2d()
        # self.fc1 = nn.Linear(?,?)
        # self.fc1 = nn.Linear(?, classes)
        # softmax
    
    def forward(self, x):
        x = self.c1(x)
        x = f.max_pool2d(x, 2)
        x = f.relu(x)
        x = self.c2(x)
        x = f.max_pool2d(x, 2)
        x = f.relu(x)
        x = f.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = f.softmax(x)

        return x


class inceptionNet(nn.Module):
    def __init__(self, classes):
        super(inceptionNet, self).__init__()
        # conv layers
        # inception
        # classifier
    
    def forward(self, x):
        pass
        


# class ???(nn.Module):
#     pass