import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xgb
from kmeans import create_model, get_deep_features, train_model
from siamese import SiameseDataset, SiameseLoss, SiameseNetwork
from sklearn.manifold import MDS
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

Current_Best_Sum_Score = [1059861.98, 1476891.76, 1658852.85]
Current_Best_Mean_Score = [52.99, 73.84, 82.94]


# This funcation calculates the positions of all channels, should be implemented by the participants
def calcLoc(
    H,
    anch_pos,
    bs_pos,
    tol_samp_num,
    anch_samp_num,
    port_num,
    ant_num,
    sc_num,
    kmeans_features=False,
    method: str = "KNN",
):
    """
    Basic implementation of channel-based localization using K-Nearest Neighbors

    Args:
        H: Complex channel data of shape (num_samples, ant_num, sc_num, port_num)
        anch_pos: Anchor positions array with columns [index, x, y]
        bs_pos: Base station position [x, y, z]
        tol_samp_num: Total number of samples
        anch_samp_num: Number of anchor samples
        port_num: Number of UE ports
        ant_num: Number of BS antennas
        sc_num: Number of subcarriers

    Returns:
        Predicted positions array of shape (tol_samp_num, 2)
    """
    # Create result array
    loc_result = np.zeros([tol_samp_num, 2], "float")

    # Extract features from channel data
    def extract_features(H_data):
        # Calculate channel magnitude
        H_mag = np.abs(H_data)

        # Extract basic statistical features
        features = []
        for i in range(H_data.shape[0]):
            sample_features = []
            # Mean over different dimensions
            sample_features.extend(
                [
                    np.mean(H_mag[i]),  # Overall mean
                    np.median(H_mag[i]),  # Overall median
                    np.std(H_mag[i]),  # Overall std
                    np.max(H_mag[i]),  # Max magnitude
                    np.min(H_mag[i]),  # Min magnitude
                ]
            )

            # Add mean per antenna
            ant_means = np.mean(H_mag[i], axis=(1, 2))
            sample_features.extend(ant_means)

            features.append(sample_features)

        return np.array(features)

    def extract_features_kmeans(H_data):
        """
        Extract features from channel data using KMeans clustering.

        Args:
            H_data: Complex channel data of shape (num_samples, ant_num, sc_num, port_num)

        Returns:
            Deep features extracted from the KMeans model.
        """
        try:
            # Reshape H_data to fit the KMeans model
            H_mag = np.abs(H_data)
            reshaped_data = H_mag.reshape(H_mag.shape[0], -1)

            # Create and train the KMeans model
            kmeans_model = create_model(n_clusters=256)
            train_model(kmeans_model, reshaped_data)

            # Extract deep features
            deep_features = get_deep_features(kmeans_model, reshaped_data)

            return deep_features
        except Exception as e:
            print(f"An error occurred during KMeans feature extraction: {e}")
            return None

    # Extract features from available channel data
    print("Extracting features...")
    X = extract_features(H) if not kmeans_features else extract_features_kmeans(H)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Prepare training data from anchor points that are within our current slice
    valid_anchors = []
    valid_positions = []

    for anchor in anch_pos:
        idx = int(anchor[0]) - 1  # Convert to 0-based index
        if idx < len(H):  # Only use anchors that are in our current slice
            valid_anchors.append(idx)
            valid_positions.append(anchor[1:])

    if len(valid_anchors) > 0:
        print(f"Training model with {len(valid_anchors)} anchor points...")
        X_train = X_scaled[valid_anchors]
        y_train = np.array(valid_positions)
        if method == "Siamese":
            input_dim = X_scaled.shape[1]
            learning_rate = 0.0003
            num_epochs = 20_000
            dataset = SiameseDataset(X_scaled, valid_anchors, y_train)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
            model = SiameseNetwork(input_dim)
            model.to("cuda")
            criterion = SiameseLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in (pbar := tqdm(range(num_epochs))):
                model.train()
                total_loss = 0
                for x1, x2, d_ij in dataloader:
                    optimizer.zero_grad()

                    z1, z2, _ = model(x1, x2)
                    loss = criterion(z1, z2, d_ij)
                    loss.backward()

                    optimizer.step()
                    total_loss += loss.item()
                pbar.set_description(
                    f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}"
                )

            with torch.no_grad():
                pairs = create_pairs(X_scaled)
                predictions = []
                for x1, x2 in pairs:
                    x1_tensor = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)
                    x2_tensor = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)

                    z1, z2, output = model(x1_tensor.to("cuda"), x2_tensor.to("cuda"))
                    predictions.append(output.cpu().numpy())

                predictions = np.array(predictions)
            torch.save(model.state_dict(), "siamese_model_state_dict.pth")
        elif method == "XGBoost":
            xgb_model = xgb.XGBRegressor(n_estimators=10, max_depth=5)
            xgb_model.fit(X_train, y_train)
            predictions = xgb_model.predict(X_scaled)
        elif method == "MDS":
            mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42)
            predictions = mds.fit_transform(X_scaled)
        elif method == "KNN":
            # Train KNN model
            knn = KNeighborsRegressor(
                n_neighbors=min(20, len(valid_anchors)), weights="distance"
            )
            knn.fit(X_train, y_train)

            # Predict positions for the current slice
            predictions = knn.predict(X_scaled)
        else:
            raise ValueError(f"Invalid method: {method}")

        # Fill the corresponding positions in the result array
        for i in range(len(H)):
            loc_result[i] = predictions[i]

    return loc_result


def plot_distance_distribution(
    prediction_file: str, ground_truth_file: str, save_path: str = None
):
    """
    Args:
        prediction_file: Path to the file containing predicted positions
        ground_truth_file: Path to the file containing ground truth positions
        save_path: Optional path to save the plot
    """

    predictions = np.loadtxt(prediction_file)
    ground_truth = np.loadtxt(ground_truth_file)

    distances = np.sqrt(np.sum((predictions - ground_truth) ** 2, axis=1))

    plt.figure(figsize=(10, 6))

    plt.hist(distances, bins=50, alpha=0.75)
    plt.axvline(
        np.mean(distances),
        color="r",
        linestyle="dashed",
        label=f"Mean Error: {np.mean(distances):.2f}m",
    )

    plt.xlabel("Distance Error (meters)")
    plt.ylabel("Number of Points")
    plt.title("Distribution of Distance Errors")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def evaluate_score(
    prediction_file: str, ground_truth_file: str, dataset_ind: str
) -> float:
    """
    Calculate score as sum of Euclidean distances between predicted and ground truth points.

    Args:
        prediction_file: Path to the file containing predicted positions (x, y)
        ground_truth_file: Path to the file containing ground truth positions (x, y)
        dataset_ind: Index of the dataset (1, 2, 3)

    Returns:
        Total score (lower is better)
    """

    dataset_ind = int(dataset_ind) - 1

    predictions = np.loadtxt(prediction_file)
    ground_truth = np.loadtxt(ground_truth_file)

    distances = np.sqrt(np.sum((predictions - ground_truth) ** 2, axis=1))

    total_score = np.sum(distances)

    mean_distance = np.mean(distances)

    print(f"\n=== Best Results ===")
    print(f"Mean distance per point: {Current_Best_Mean_Score[dataset_ind]:.2f} meters")
    print(f"Number of points evaluated: {len(distances)}")
    print("========================")

    print(f"\n=== Evaluation Results ===")
    print(f"Mean distance per point: {mean_distance:.2f} meters")
    print(f"Number of points evaluated: {len(distances)}")
    print("========================")

    return total_score


def create_pairs(X_scaled):
    pairs = []
    num_samples = len(X_scaled)
    for i in range(num_samples):
        # Randomly select another sample to form a pair with the i-th sample
        j = np.random.choice([x for x in range(num_samples) if x != i])
        pairs.append((X_scaled[i], X_scaled[j]))  # Create a pair of (x1, x2)
    return pairs
