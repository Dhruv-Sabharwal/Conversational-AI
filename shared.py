import torch
import torch.nn as nn

class SharedBiLSTM(nn.Module):
    def __init__(self, input_size=256, hidden_size=256, batch_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size//2, bidirectional=True)
    
    def forward(self, x):
        h0 = torch.randn(2, self.batch_size, self.hidden_size//2).to('cuda')  # hidden state
        c0 = torch.randn(2, self.batch_size, self.hidden_size//2).to('cuda')  # cell state
        out, _ = self.lstm(x, (h0, c0))
        return out