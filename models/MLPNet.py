# coding:utf8
from torch import nn
from torch.nn import functional as F
from .BasicModule import BasicModule


class MLPNet(BasicModule):

    def __init__(self):
        super(MLPNet, self).__init__()

        self.model_name = 'MLPNet'

        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 10)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x
