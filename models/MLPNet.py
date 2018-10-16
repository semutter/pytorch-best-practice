# coding:utf8
import torch as t
from torch import nn
from torch.nn import functional as F
from .BasicModule import BasicModule


class MLPNet(BasicModule):

    def __init__(self):
        super(MLPNet, self).__init__()

        self.model_name = 'MLPNet'

        num = 30
        self.fc1 = nn.Linear(28 * 28, num)
        self.fc2 = nn.Linear(num, 10)
        t.nn.init.normal_(self.fc1.weight, 100, 100)
        t.nn.init.normal_(self.fc2.weight, 100, 100)


    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.sigmoid(self.fc1(x))
        return self.fc2(x)
