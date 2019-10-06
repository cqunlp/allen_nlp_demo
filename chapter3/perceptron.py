import torch
import torch.nn as nn

class Perceptron(nn.Module):
    """A perceptron is one linear layer"""

    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fcl = nn.Linear(input_dim,1)
    
    def forward(self, x_in):
        return torch.sigmoid(self.fcl(x_in)).squeeze()

