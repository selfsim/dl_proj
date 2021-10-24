# This class contains my model architectures
# It contains the following:
# a linear classifier with no hidden layers
# a convulutional neural network with ...
# a DNN with inception type architectures
# ???
from random import seed
from random import random
import torch.nn as nn

class linearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(linearClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class convNet(nn.Module):
    pass

class inceptionNet(nn.Module):
    pass