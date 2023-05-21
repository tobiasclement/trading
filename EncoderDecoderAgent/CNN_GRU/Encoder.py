import torch
import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, number_of_index, window_size, hidden_size, device):
        """

        :param state_size: we give OHLC as input to the network
        """
        super(Encoder, self).__init__()
        self.number_of_index = number_of_index
        self.hidden_size = hidden_size
        self.device = device

        self.conv_encoder = nn.Sequential(
            nn.Conv1d(window_size, window_size, 3),
            nn.Conv1d(window_size, window_size, 2)
        )
        #self.conv_encoder = nn.Sequential(
            #nn.Conv1d(window_size, window_size, 3),
            #nn.Conv1d(window_size, window_size, 2)
        #)

        #Tobias: changed shape first dimension from 1 to 0
        if number_of_index == 2:
            self.gru = nn.GRU(9, hidden_size)
        elif number_of_index == 1:
            self.gru = nn.GRU(5, hidden_size)
        else: self.gru = nn.GRU(1, hidden_size)
        #Tobias: changed shape first dimension from 1 to 0
        #self.index = 1
        #self.gru = nn.GRU(5, hidden_size)

    def forward(self, x):
        """

        :param x: input is of type [window_size, batch_size, input_size]
        :return:
        """
        
        # We have to convolve over time axis, thus we change the column of time and price
        hidden = self.initHidden(x.shape[1])

        x = x.permute(1, 0, 2)
 
        conv_out = self.conv_encoder(x)
        output, hidden = self.gru(conv_out.permute(1, 0, 2), hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=self.device)
