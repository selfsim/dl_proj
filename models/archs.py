import torch
import torch.nn as nn
import torchvision.models as models

nn_models = ['linear', 'conv', 'google']

class linearClassifier(nn.Module):
    def __init__(self, input_dim, classes):
        super(linearClassifier, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=classes)

    def forward(self, x):
        return self.linear(x)

class convNet(nn.Module):
    def __init__(self, classes):
        super(convNet, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 64, 7, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,1),
            nn.Conv2d(64, 128, 5, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,1),
            nn.Conv2d(128, 256, 5, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,1),
            nn.Dropout()
        )

        self.linear_stack = nn.Sequential(
            nn.Linear(57600, 1024),
            nn.Linear(1024, classes),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = x.view(x.size(0), -1)
        x = self.linear_stack(x)
        return x


class googleNet(models.GoogLeNet):
    def __init__(self, classes):
        super(googleNet, self).__init__(classes, aux_logits=False)

