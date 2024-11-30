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
        Generate a data pair (x1, x2) and get the ground truth position of x1.
        """
        i = self.valid_anchors[idx]
        x1 = self.X[i]

        # Randomly sample a positive or negative pair
        if np.random.rand() > 0.5:  # 50% chance for positive or negative pair
            j = idx  # Positive pair (self-comparison)
        else:
            j = np.random.choice([k for k in range(len(self.valid_anchors)) if k != idx])  # Negative pair

        x2 = self.X[self.valid_anchors[j]]

        B = 64
        N = np.sqrt(0.5) * (np.random.randn(x1.shape[0]))
        x1 += N

        real_y1 = self.valid_positions[idx]

        return (
            torch.tensor(x1, dtype=torch.float32).to(self.device),
            torch.tensor(x2, dtype=torch.float32).to(self.device),
            torch.tensor(real_y1, dtype=torch.float32).to(self.device),
        )

import torch
import torch.nn as nn
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self, device="cuda"):
        """
        Initialize the FeatureExtractor.
        Args:
            n_ue_ant (int): Number of UE antennas.
            n_bs_ant (int): Number of BS antennas.
            noise_std (float): Standard deviation of Gaussian noise.
            device (str): Device to use for computation (e.g., 'cuda' or 'cpu').
        """
        super(FeatureExtractor, self).__init__()
        self.device = device
        self.training = False

    def forward(self, csi_data):
        """
        Forward pass to extract features from CSI data.
        Args:
            csi_data (torch.Tensor): Input CSI data of shape (batch_size, n_ue_ant, n_bs_ant, n_subcarriers).
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, feature_dim).
        """
        # Step 1: Add Gaussian noise (only during training)

        n_samples, n_ue_ant, n_bs_ant, n_subcarriers = csi_data.shape
        if self.training:
            noise = np.sqrt(0.5) * (np.random.randn(*csi_data.shape))
            csi_data = csi_data + noise

        # Step 2: Compute Frobenius norm and normalize
        fro_norms = torch.sqrt(torch.sum(torch.abs(csi_data) ** 2, dim=(1, 2, 3)))
        csi_data = csi_data / fro_norms[:, np.newaxis, np.newaxis, np.newaxis]

        # Step 3: Scale by the scaling factor

        scaling_factor = np.sqrt(n_ue_ant * n_bs_ant)
        csi_data = csi_data * scaling_factor

        # Step 4: Compute 2D Fourier Transform along UE and BS antennas
        beamspace_data = torch.fft.fft2(csi_data, dim=(1, 2)) / scaling_factor

        # Step 5: Compute magnitude of the beamspace data
        beamspace_magnitudes = torch.abs(beamspace_data)

        # Step 6: Flatten the beamspace magnitudes
        features = beamspace_magnitudes.view(beamspace_magnitudes.size(0), -1)

        return features

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
        self.feature_extractor = FeatureExtractor()

    def forward(self, x):
        x = self.feature_extractor(x)
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

        return output1, output2


class SiameseLoss(nn.Module):
    def __init__(self):
        """
        Initialize the Siamese Loss function
        """

        super(SiameseLoss, self).__init__()

    def forward(self, x1, x2, y1, y2, yp1):
        """
        Compute the Siamese loss based on https://arxiv.org/pdf/1909.13355
        Args:
            x1, x2: Higher dimensional input features
            y1, y2: Low Dimension results of model from input features
            yp1: Ground truth for x1
        """
        distance_x = torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1) + 1e-6)
        distance_y = torch.sqrt(torch.sum((y1 - y2) ** 2, dim=1) + 1e-6)

        param_w = torch.ones(x1.shape[0]).to(x1.device)

        for i in range(x1.shape[0]):
            if not torch.equal(x1[i], x2[i]):
                param_w[i] = distance_x[i]

        loss = torch.sum(((distance_x - distance_y) ** 2) / param_w)

        distance_gt = torch.sum((y1 - yp1) ** 2, dim=1)
        loss_gt = torch.sum(distance_gt)

        return loss + loss_gt
