##___________________---------- 3 layers with dropout---------------___________
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




# ##_____________ four layer of NN______________
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SiameseNetwork(nn.Module):
#     def __init__(self, input_dim):
#         """
#         Initialize the Siamese Network with a given input dimension.

#         Args:
#             input_dim (int): Dimension of the input features.
#         """
#         super(SiameseNetwork, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 32)
#         self.fc4 = nn.Linear(32, 16)  # Added fc4 layer

#     def forward_one(self, x):
#         """
#         Forward pass for a single branch of the Siamese Network.

#         Args:
#             x (Tensor): Input tensor.

#         Returns:
#             Tensor: Output tensor after passing through the network layers.
#         """
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)  # Now fc4 exists
#         return x

#     def forward(self, input1, input2):
#         """
#         Forward pass for the Siamese Network comparing two inputs.

#         Args:
#             input1 (Tensor): First input tensor.
#             input2 (Tensor): Second input tensor.

#         Returns:
#             Tensor: The pairwise distance between the two outputs.
#         """
#         output1 = self.forward_one(input1)
#         output2 = self.forward_one(input2)
#         return F.pairwise_distance(output1, output2)
