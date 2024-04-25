import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreModel(nn.Module):
    def __init__(self):
        super(ScoreModel, self).__init__()

        self.shared_mlp = nn.Sequential(
            nn.Linear(512, 128), 
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid() 
        )

    def forward(self, input):

        score = self.shared_mlp(input)

        return score
    