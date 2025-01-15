import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import sys

sys.path.append("../..")
from Game.BlackJack import BlackJack




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

class nnAgent:
    def __init__(self, model = "Agent/nn/blackjack_nn_model.pth"):
        self.model = BlackjackNN()
        self.model.load_state_dict(torch.load(model))
        self.model.eval()
        
    
    def encode_input(self, game: BlackJack):
        player_sum = game.get_playervalue()
        dealer_card_num = game.total_value(game.dealer_hand[:1])
        card_count = game.card_count
        has_ace = 0
        for card in game.player_hand:
            if card["number"] == "A":
                has_ace = 1
                break
        input_array = np.array([player_sum, has_ace, 
                            dealer_card_num]).reshape(1,-1)
        cc_array = pd.DataFrame.from_dict([card_count])

        input_array = np.concatenate([input_array, cc_array], axis=1)
        input_array = np.delete(input_array, -1, axis=1)
        input_array = torch.tensor(input_array, dtype=torch.float32)
        return input_array
    
    
    def choose_action(self, game: BlackJack):
        input_array = self.encode_input(game)
        output = self.model(input_array)
        predict = F.softmax(output, dim=1)
        predict = torch.argmax(predict, axis=1)
        if predict == 0:
            return "stay"
        else:
            return "hit"