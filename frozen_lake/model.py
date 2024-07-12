from torch import nn
import torch.nn.functional as F

class DeepQLearningNetwork(nn.Module):
    def __init__(self, input_state, hidden_layer, output_state):
        super().__init__()
        self.first_layer = nn.Linear(input_state, hidden_layer)  
        self.last_layer = nn.Linear(hidden_layer, output_state) 

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        x = self.last_layer(x)        
        return x