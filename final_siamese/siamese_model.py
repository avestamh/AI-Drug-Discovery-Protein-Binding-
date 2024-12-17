'''
This file defines the Siamese Neural Network architecture:
The network consists of:
Three fully connected layers.
Batch Normalization and Dropout for regularization.
Contrastive loss to compute the distance between embeddings.
The model takes two feature vectors as input and outputs a similarity distance.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x1, x2):
        x1 = F.relu(self.bn1(self.fc1(x1)))
        x1 = self.dropout(x1)
        x1 = F.relu(self.bn2(self.fc2(x1)))
        
        x2 = F.relu(self.bn1(self.fc1(x2)))
        x2 = self.dropout(x2)
        x2 = F.relu(self.bn2(self.fc2(x2)))

        distance = torch.abs(x1 - x2)
        output = self.fc3(distance)
        return output

