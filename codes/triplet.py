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

        self.calculated_X = []

        for i, _ in tqdm(enumerate(self.X), total=len(self.X)):
            self.calculated_X.append(self.calculate_features(self.X[i]))

        for i, _ in tqdm(enumerate(self.X), total=len(self.X)):
            y_anchor = self.valid_positions[i]
            for j, x in enumerate(self.X):
                y = self.valid_positions[j]
                if not i in self.distances:
                    self.distances[i] = []
                self.distances[i].append((np.sqrt(np.sum(y_anchor - y) ** 2), i, x))
            
            self.distances[i].sort(key=lambda x: x[0])

    def calculate_features(self, x, withNoise=False):
        n_ue_ant, n_bs_ant, n_subcarriers = x.shape

        signal_power = np.mean(np.abs(x) ** 2)

        snr_db = 0
        snr_linear = 10 ** (snr_db / 10)

        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power / 2), size=x.shape) + 1j * np.random.normal(0, np.sqrt(noise_power / 2), size=x.shape)

        if withNoise == True: 
            x = x + noise

        fro_norms = np.sqrt(np.sum(np.abs(x) ** 2))
        x = x / fro_norms


        scaling_factor = np.sqrt(n_ue_ant * n_bs_ant)
        x = x * scaling_factor

        beamspace_batch = np.fft.fft2(x, axes=(0,1))

        # print(f"beamspace_batch: {beamspace_batch.shape}")

        beamspace_batch /= np.sqrt(n_ue_ant * n_bs_ant)  # Normalize by number of antennas

        # print(f"beamspace_batch v2: {beamspace_batch.shape}")

        x = np.abs(beamspace_batch)

        # print(f"beamspace_batch v3: {beamspace_batch.shape}")
# beamspace_magnitudes.reshape(len(csi_data), -1)
        return x.reshape(-1)


    def __len__(self):
        return len(self.valid_anchors)

    def __getitem__(self, idx):

        normal = np.random.rand()
        new_noise = False
    
        if (normal < 0.25):
            # print(normal)
            new_noise = True
        # print(idx)
        i = self.valid_anchors[idx]
        x_anchor = self.X[i]

        if new_noise:
            x_anchor = self.calculate_features(x_anchor, new_noise)
        else:
            x_anchor = self.calculated_X[i]


        y_anchor = self.valid_positions[idx]
        distance_for_x = self.distances[idx]


        x_close_id = np.random.randint(0, len(distance_for_x) // 4)
        x_close_id_in_X = distance_for_x[x_close_id][1] 
        x_close = distance_for_x[x_close_id][2]
        
        if new_noise:
            x_close = self.calculate_features(x_close, new_noise)
        else:
            x_close = self.calculated_X[x_close_id_in_X]

        x_far_idx = np.random.randint(len(distance_for_x) // 4, len(distance_for_x))
        x_far_id_in_X = distance_for_x[x_far_idx][1]
        x_far = distance_for_x[x_far_idx][2]
        
        if new_noise:
            x_far = self.calculate_features(x_far, new_noise)
        else:
            x_far = self.calculated_X[x_far_id_in_X]

       
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
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, embedding_dim)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        self.bn6 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc(x))
        # x = self.bn1(x)

        x = F.relu(self.fc1(x))
        # x = self.bn2(x)

        x = F.relu(self.fc2(x))
        # x = self.bn3(x)

        x = F.relu(self.fc3(x))
        # x = self.bn4(x)
        
        x = F.relu(self.fc4(x))
        # x = self.bn5(x)
        
        x = F.relu(self.fc5(x))
        # x = self.bn6(x)
        
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
    def __init__(self, M=0.2):
        """
        Initialize the Siamese Loss function
        """
        self.type_loss = "MarginLoss"
        self.M = M
        super(TripletLoss, self).__init__()

    def margin_loss(self, y_close, y_anchor, y_far, y_true_anchor):
        distance_close = torch.sqrt(torch.sum((y_close - y_anchor) ** 2, dim=1) + 1e-6)
        distance_far = torch.sqrt(torch.sum((y_far - y_anchor) ** 2, dim=1) + 1e-6)

        loss = torch.sum(F.relu(distance_close - distance_far + self.M)) / y_close.shape[0]

        # print(y_anchor.shape, y_true_anchor.shape)

        distance_gt = torch.sum((y_anchor - y_true_anchor) ** 2, dim=1)
        loss_gt = torch.sum(distance_gt) / y_close.shape[0]

        # loss_gt = 0

        return loss + loss_gt
    
    def exp_loss(self, y_close, y_anchor, y_far, y_true_anchor):
        distance_close = torch.sqrt(torch.sum((y_close - y_anchor) ** 2, dim=1) + 1e-6)
        distance_far = torch.sqrt(torch.sum((y_far - y_anchor) ** 2, dim=1) + 1e-6)

        loss = torch.log(torch.sum(torch.exp(distance_close - distance_far))) / 2000

        distance_gt = torch.sum((y_anchor - y_true_anchor) ** 2, dim=1)
        loss_gt = torch.sum(distance_gt) / y_close.shape[0]

        return loss + loss_gt


    def forward(self, y_close, y_anchor, y_far, y_true_anchor):

        if self.type_loss == "MarginLoss":
            return self.margin_loss(y_close, y_anchor, y_far, y_true_anchor)   
        elif self.type_loss == "ExpLoss":
            return self.exp_loss(y_close, y_anchor, y_far, y_true_anchor)
        

