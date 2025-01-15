import torch
from torch import nn, optim
import torch.nn.functional as F

class BlackjackNN(nn.Module):
    def __init__(self):
        super(BlackjackNN, self).__init__()
        self.fc1 = nn.Linear(13, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)  
        return x