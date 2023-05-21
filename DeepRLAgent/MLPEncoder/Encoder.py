import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, number_of_index, num_classes, state_size):
        """
        
        :param state_size: we give OHLC as input to the network
        :param action_length: Buy, Sell, Idle
        """
       
        super(Encoder, self).__init__()
        self.number_of_index = number_of_index
        
        #Tobias: changed shape first dimension from 1 to 0
        if number_of_index == 2: state_size=12
        elif number_of_index == 1: state_size=8
        else: state_size=4

        self.encoder = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.BatchNorm1d(128),
            # nn.Linear(128, 256),
            # nn.BatchNorm1d(256),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
