import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


class TripletDataset(Dataset):
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
        self.device = device

        self.distances = {}
        for i, _ in tqdm(enumerate(self.X), total=len(self.X)):
            y_anchor = self.valid_positions[i]
            for j, x in enumerate(self.X):
                y = self.valid_positions[j]
                if not i in self.distances:
                    self.distances[i] = []
                self.distances[i].append((np.sqrt(np.sum(y_anchor - y) ** 2), x))
            
            self.distances[i].sort(key=lambda x: x[0])


    def __len__(self):
        return len(self.valid_anchors)

    def __getitem__(self, idx):


        i = self.valid_anchors[idx]
        x_anchor = self.X[i]
        y_anchor = self.valid_positions[idx]

        distance_for_x = self.distances[idx]
        # print(distance_for_x)

        
        #         combined = []
        # for i,x in enumerate(all_x):
        #     y = self.valid_positions[i]
        #     combined.append((np.sqrt(np.sum(y_anchor - y) ** 2), x))

        # combined.sort(key=lambda x: x[0])


        x_close_idx = np.random.randint(0, len(distance_for_x)//2)
        x_close = distance_for_x[x_close_idx][1]
        x_far_idx = np.random.randint(x_close_idx, len(distance_for_x))
        x_far = distance_for_x[x_far_idx][1]

       
        return (
            torch.tensor(x_close, dtype=torch.float32).to(self.device), 
            torch.tensor(x_anchor, dtype=torch.float32).to(self.device), 
            torch.tensor(x_far, dtype=torch.float32).to(self.device), 
            torch.tensor(y_anchor, dtype=torch.float32).to(self.device), 
        )


class TripletNetworkBase(nn.Module):
    def __init__(self, input_dim, embedding_dim=2):
        """
        Initialize the Siamese Network.
        Args:
            input_dim: Dimension of input features
            embedding_dim: Dimension of output embeddings
            dropout_rate: Dropout rate for regularization
        """
        super(TripletNetworkBase, self).__init__()
        self.fc = nn.Linear(input_dim, 1024)
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, embedding_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class TripletNetwork(nn.Module):
    def __init__(self, input_dim):
        super(TripletNetwork, self).__init__()
        self.base_network = TripletNetworkBase(input_dim)

    def forward(self, input_close, input_anchor, input_far):
        output_close = self.base_network(input_close)
        output_anchor = self.base_network(input_anchor)
        output_far = self.base_network(input_far)

        return output_close, output_anchor, output_far


class TripletLoss(nn.Module):
    def __init__(self, M=1):
        """
        Initialize the Siamese Loss function
        """
        self.M = M
        super(TripletLoss, self).__init__()

    def forward(self, y_close, y_anchor, y_far, y_true_anchor):

        distance_close = torch.sqrt(torch.sum((y_close - y_anchor) ** 2, dim=1) + 1e-6)
        distance_far = torch.sqrt(torch.sum((y_far - y_anchor) ** 2, dim=1) + 1e-6)

        loss = torch.sum(F.relu(distance_close - distance_far + self.M))

        # print(y_anchor.shape, y_true_anchor.shape)
        distance_gt = torch.sum((y_anchor - y_true_anchor) ** 2, dim=1)
        loss_gt = torch.sum(distance_gt)

        return loss + loss_gt

