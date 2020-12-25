import torch
import torch.nn as nn

class IntentBiLSTM(nn.Module):
    def __init__(self, num_classes, input_size=256, hidden_size=256, batch_size=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes   # How many intents?
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size//2, bidirectional=True)
        self.hidden2intent = nn.Linear(self.hidden_size, self.num_classes)
        self.softmax = nn.Softmax(dim=-1)
        
    
    def forward(self, x):
        h0 = torch.randn(2, self.batch_size, self.hidden_size//2).to('cuda')  # hidden state
        c0 = torch.randn(2, self.batch_size, self.hidden_size//2).to('cuda')  # cell state
        out, _ = self.lstm(x, (h0, c0))
        out = self.hidden2intent(out[-1, :, :])
        out = self.softmax(out)
        return out
    
    
def get_predicted_intent(a):
    biggest = max(a)
    for i in range(len(a)):
        if a[i] == biggest:
            a[i] = 1.
        else:
            a[i] = 0.
    return a


# This function returns the intent classification accuracy (either 0 or 1 since it calculates for only one sample).
def get_ICA(pred, gt):
    pred = get_predicted_intent(pred)
    if(torch.equal(pred, gt)):
        return 1.
    else:
        return 0.