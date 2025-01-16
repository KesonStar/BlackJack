import torch
from torch import nn, optim
import torch.nn.functional as F

class BlackjackNN(nn.Module):
    def __init__(self, D = 8, W = 256, input_ch = 12, output_ch = 2, skips = [4]):
        super(BlackjackNN, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        layers = [nn.Linear(input_ch, W)]
        
        for i in range(D - 1):
            layer = nn.Linear

            in_channels = W
            if i in self.skips:
                in_channels += input_ch

            layers += [layer(in_channels, W)]
            
        self.layers = nn.ModuleList(layers)
        self.output_linear = nn.Linear(W, output_ch)
        
        

    def forward(self, x):
        input_x = x
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))
            if i in self.skips:
                x = torch.cat([x, input_x], dim=1)
        return self.output_linear(x)