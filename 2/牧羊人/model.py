import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#  Hyperparameters
BATCH_SIZE = 32
LR = 0.01                   # 学习率
EPSILON = 0.9               # e-greedy
GAMMA = 0.9                 # 衰减参数
TARGET_REPLACE_ITER = 4   # Q 现实网络的更新频率100次循环更新一次
MEMORY_CAPACITY = 2000      # 记忆库大小
N_ACTIONS = 7
N_STATES = 1


class Net(nn.Module):
    def __init__(self, num_actions):
        super(Net, self).__init__()
        self.p1 = nn.MaxPool2d(3)
        self.f1 = nn.Linear(49, 16)
        self.f1.weight.data.normal_(0, 0.1)
        self.f2 = nn.Linear(16, num_actions)
        self.f2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.p1(x)
        x = x.view(x.size(0), -1)
        x = self.f1(x)
        x = F.relu(x)
        action = self.f2(x)
        return action

