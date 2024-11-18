import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class SiameseDataset(Dataset):
    def __init__(self, X, valid_anchors, valid_positions, device="cuda"):
        """
        Initialize the dataset with input features, valid anchor indices, and their positions.
        Args:
            X: Input features (numpy array)
            valid_anchors: List of indices for valid anchors
            valid_positions: List of positions corresponding to valid anchors
        """
        self.X = X
        self.valid_anchors = valid_anchors
        self.valid_positions = valid_positions
        self.valid_anchors = self.valid_anchors
        self.device = device

    def __len__(self):
        return len(self.valid_anchors)

    def __getitem__(self, idx):
        """
        Generate a data pair (x1, x2) and compute the distance between their ground truth positions.
        """
        i = self.valid_anchors[idx]
        x1 = self.X[i]

        # Randomly sample a positive or negative pair
        if np.random.rand() > 0.5:  # 50% chance for positive or negative pair
            j = idx  # Positive pair (self-comparison)
        else:
            j = np.random.choice(
                [k for k in range(len(self.valid_anchors)) if k != idx]
            )  # Negative pair

        x2 = self.X[self.valid_anchors[j]]

        real_x1 = self.valid_positions[idx]
        # d_ij = np.linalg.norm(
        #     self.valid_positions[idx] - self.valid_positions[j]
        # )  # Euclidean distance
        return (
            torch.tensor(x1, dtype=torch.float32).to(self.device),
            torch.tensor(x2, dtype=torch.float32).to(self.device),
            torch.tensor(real_x1, dtype=torch.float32).to(self.device),
        )


class SiameseNetworkBase(nn.Module):
    def __init__(self, input_dim, embedding_dim=2):
        """
        Initialize the Siamese Network.
        Args:
            input_dim: Dimension of input features
            embedding_dim: Dimension of output embeddings
            dropout_rate: Dropout rate for regularization
        """
        super(SiameseNetworkBase, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, embedding_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SiameseNetwork, self).__init__()
        self.base_network = SiameseNetworkBase(input_dim)

    def forward(self, input1, input2):
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)

        distance = torch.abs(output1 - output2)

        output = torch.sigmoid(distance)
        return output1, output2, output


class SiameseLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Initialize the Siamese Loss function with an optional margin.
        Args:
            margin: Margin for contrastive loss
        """
        super(SiameseLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y1, y2, yp1):
        """
        Compute the Siamese loss based on Euclidean distances.
        Args:
            x1, x2: Higher dimensional input features
            y1, y2: Low Dimension results of model from input features
            yp1: Ground truth for x1
        """
        distance_x = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1) + 1e-6)
        distance_y = torch.sqrt(torch.sum((y1 - y2) ** 2, dim=1) + 1e-6)

        param = 1
        if(not torch.equal(x1,x2)):
            param = distance_x

        loss = torch.sum(((distance_x - distance_y) ** 2) / param)

        distance_gt = torch.sum((y1 - yp1) ** 2, dim=1) 
        loss_gt = torch.sum(distance_gt)
        
        # print(loss, loss_gt)

        return loss + loss_gt
    
        # distance = torch.sqrt(torch.sum((z1 - z2) ** 2, dim=1) + 1e-6)
        # loss = torch.mean((d_ij - distance) ** 2)
        # margin_loss = torch.clamp(self.margin - distance, min=0)
        # return loss + torch.mean(margin_loss)