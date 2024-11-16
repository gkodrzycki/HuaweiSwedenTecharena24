import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Dataset class for Siamese Network
class SiameseDataset(Dataset):
    def __init__(self, X, valid_anchors, valid_positions, device="cuda", threshold=1.0):
        """
        Initialize the dataset with input features, valid anchor indices, and their positions.
        Args:
            X: Input features (numpy array)
            valid_anchors: List of indices for valid anchors
            valid_positions: List of positions corresponding to valid anchors
            device: Device for PyTorch tensors
            threshold: Threshold distance for determining similarity
        """
        self.X = X
        self.valid_anchors = valid_anchors
        self.valid_positions = valid_positions
        self.device = device
        self.threshold = threshold

    def __len__(self):
        return len(self.valid_anchors)

    def __getitem__(self, idx):
        """
        Generate a data pair (x1, x2) and determine if it's a positive or negative pair.
        """
        i = self.valid_anchors[idx]
        x1 = self.X[i]

        # Generate a random pair (positive or negative)
        if np.random.rand() > 0.5:  # Positive pair
            j = idx
            y_true = 1  # Similar
        else:  # Negative pair
            j = np.random.choice([k for k in range(len(self.valid_anchors)) if k != idx])
            y_true = 0  # Dissimilar

        x2 = self.X[self.valid_anchors[j]]

        # Return data tensors
        return (
            torch.tensor(x1, dtype=torch.float32).to(self.device),
            torch.tensor(x2, dtype=torch.float32).to(self.device),
            torch.tensor(y_true, dtype=torch.float32).to(self.device),
        )


# Base network for Siamese Network
class SiameseNetworkBase(nn.Module):
    def __init__(self, input_dim, embedding_dim=2):
        """
        Initialize the base network for embedding extraction.
        Args:
            input_dim: Dimension of input features
            embedding_dim: Dimension of output embeddings
        """
        super(SiameseNetworkBase, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, embedding_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Full Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize the Siamese Network.
        Args:
            input_dim: Dimension of input features
        """
        super(SiameseNetwork, self).__init__()
        self.base_network = SiameseNetworkBase(input_dim)

    def forward(self, input1, input2):
        # Get embeddings for both inputs
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)

        # Compute Euclidean distance
        distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1) + 1e-6)
        return output1, output2, distance


# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        """
        Initialize the contrastive loss function.
        Args:
            margin: Margin for dissimilar pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, z1, z2, y_true):
        """
        Compute contrastive loss.
        Args:
            z1, z2: Embeddings of the input pairs
            y_true: Binary label (1 for similar, 0 for dissimilar)
        """
        distance_square = torch.sum((z1 - z2) ** 2, dim=1)
        positive_loss = y_true * distance_square  # Loss for similar pairs
        negative_loss = (1 - y_true) * torch.clamp(self.margin - torch.sqrt(distance_square + 1e-6), min=0) ** 2
        return torch.mean(positive_loss + negative_loss)


# Training Loop
def train_siamese_network(X_scaled, valid_anchors, y_train, learning_rate=0.0003, num_epochs=1000, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare dataset and dataloader
    dataset = SiameseDataset(X_scaled, valid_anchors, y_train, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    input_dim = X_scaled.shape[1]
    model = SiameseNetwork(input_dim).to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in (pbar := tqdm(range(num_epochs))):
        model.train()
        total_loss = 0

        for x1, x2, y_true in dataloader:
            optimizer.zero_grad()

            z1, z2, distance = model(x1, x2)
            loss = criterion(z1, z2, y_true)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()

        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}"
        )

    return model