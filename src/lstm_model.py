import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input):
        _, (h_n, _) = self.lstm(input)
        classification = self.linear(h_n[-1])

        return classification