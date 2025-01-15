import numpy
import pandas as pd
import torch
from torch import nn, optim

def data_preprocessing(data_dir):
    df = pd.read_csv(data_dir)
    
    feature_list = [i for i in df.columns if i not in ['dealer_card',
                                                         'Y','lose',
                                                         'correct_action',
                                                         'dealer_bust',
                                                         'dealer_bust_pred',
                                                         'new_stack', 'games_played_with_stack',
                                                         2,3,4,5,6,7,8,9,10,'A',
                                                        ]]
    X = torch.tensor(df[feature_list].values).float()
    Y = torch.tensor(df['correct_action'].values).long()
    return X, Y


def set_training_params(model, learning_rate=1e-3):
    """
    设置训练相关的优化器和损失函数。

    Args:
        model (nn.Module): 需要训练的模型。
        learning_rate (float): 学习率，默认 1e-3。

    Returns:
        criterion (nn.CrossEntropyLoss): 交叉熵损失函数。
        optimizer (torch.optim.Optimizer): Adam 优化器。
    """
    loss = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    return loss, optimizer
    