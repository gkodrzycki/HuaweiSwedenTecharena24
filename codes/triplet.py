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

        # self.distances = {}
        # for i, _ in tqdm(enumerate(self.X), total=len(self.X)):
        #     y_anchor = self.valid_positions[i]
        #     for j, x in enumerate(self.X):
        #         y = self.valid_positions[j]
        #         if not i in self.distances:
        #             self.distances[i] = []
        #         self.distances[i].append((np.sqrt(np.sum(y_anchor - y) ** 2), x))
            
        #     self.distances[i].sort(key=lambda x: x[0])
        self.position_distances = torch.cdist(self.valid_positions, self.valid_positions)

        self.distance_ranking = {}
        for i in range(len(X)):
            # Sort indices by position distance, excluding self
            sorted_indices = torch.argsort(self.position_distances[i])
            self.distance_ranking[i] = sorted_indices[1:]


    def __len__(self):
        return len(self.valid_anchors)

    # def __getitem__(self, idx):


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

        signal_power = np.mean(x_anchor ** 2)

        snr_db = 0
        snr_linear = 10 ** (snr_db / 10)

        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power / 2), size=x_anchor[i].shape)
        x_close = x_anchor + noise

        x_close_id = np.random.randint(0, len(distance_for_x) // 4)
        x_close = distance_for_x[x_close_id][1] 
        x_far_idx = np.random.randint(len(distance_for_x) // 4, len(distance_for_x))
        x_far = distance_for_x[x_far_idx][1]

       
        return (
            torch.tensor(x_close, dtype=torch.float32).to(self.device), 
            torch.tensor(x_anchor, dtype=torch.float32).to(self.device), 
            torch.tensor(x_far, dtype=torch.float32).to(self.device), 
            torch.tensor(y_anchor, dtype=torch.float32).to(self.device), 
        )

    def __getitem__(self, idx):
        x_anchor = self.X[idx].to(self.device)
        y_anchor = self.valid_positions[idx].to(self.device)

        # Sampling strategy based on position distances
        ranked_indices = self.distance_ranking[idx]
        
        # Select a close point (first few in the ranking)
        close_idx = ranked_indices[np.random.randint(0, min(5, len(ranked_indices)))].item()
        x_close = self.X[close_idx].to(self.device)
        
        # Select a far point (from the last part of the ranking)
        far_idx = ranked_indices[np.random.randint(len(ranked_indices) - 5, len(ranked_indices))].item()
        x_far = self.X[far_idx].to(self.device)

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

    def forward(self, x):
        x = F.relu(self.fc(x))

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
    def __init__(self, M=0.2):
        """
        Initialize the Triple Loss function with added adaptivity
        """
        self.type_loss = "MarginLoss"
        self.M = M
        super(TripletLoss, self).__init__()

    def margin_loss(self, y_close, y_anchor, y_far, y_true_anchor):
        # Compute Euclidean distances with numerical stability
        distance_close = torch.sqrt(torch.sum((y_close - y_anchor) ** 2, dim=1) + 1e-6)
        distance_far = torch.sqrt(torch.sum((y_far - y_anchor) ** 2, dim=1) + 1e-6)

        # Margin-based triplet loss
        loss = torch.sum(F.relu(distance_close - distance_far + self.M)) / y_close.shape[0]

        # Ground truth positioning loss
        distance_gt = torch.sum((y_anchor - y_true_anchor) ** 2, dim=1)
        loss_gt = torch.sum(distance_gt) / y_close.shape[0]

        return loss + 0.1 * loss_gt
    
    def exp_loss(self, y_close, y_anchor, y_far, y_true_anchor):
        # Exponential loss variant
        distance_close = torch.sqrt(torch.sum((y_close - y_anchor) ** 2, dim=1) + 1e-6)
        distance_far = torch.sqrt(torch.sum((y_far - y_anchor) ** 2, dim=1) + 1e-6)

        # Logarithmic loss to handle exponential distances
        loss = torch.log(torch.sum(torch.exp(distance_close - distance_far))) / 2000

        return loss

    def forward(self, y_close, y_anchor, y_far, y_true_anchor):
        # Flexible loss computation
        if self.type_loss == "MarginLoss":
            return self.margin_loss(y_close, y_anchor, y_far, y_true_anchor)   
        elif self.type_loss == "ExpLoss":
            return self.exp_loss(y_close, y_anchor, y_far, y_true_anchor)


class SemiSupervisedLoss(torch.nn.Module):
    def __init__(self, threshold=0.9):
        super(SemiSupervisedLoss, self).__init__()
        self.threshold = threshold

    def forward(self, predictions, true_labels=None):
        """
        Args:
            predictions: Predicted logits from the model (B, C).
            true_labels: Ground truth labels for the labeled data (B,) or None.
        """
        # Supervised loss for labeled data
        supervised_loss = 0
        if true_labels is not None:
            supervised_loss = F.cross_entropy(predictions, true_labels)

        # Pseudo-labeling for unlabeled data
        with torch.no_grad():
            probabilities = F.softmax(predictions, dim=1)
            pseudo_labels = torch.argmax(probabilities, dim=1)
            max_probs = torch.max(probabilities, dim=1)[0]

        # Mask for confident predictions
        mask = max_probs > self.threshold
        pseudo_loss = F.cross_entropy(predictions[mask], pseudo_labels[mask]) if mask.any() else 0

        return supervised_loss + pseudo_loss

