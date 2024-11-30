from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xgb
from siamese import SiameseDataset, SiameseLoss, SiameseNetwork
from torch.utils.data import DataLoader
from tqdm import tqdm

Current_Best_Mean_Score = [10.78, 14.07, 34.43]


def extract_features(csi_data):

    n_samples, n_ue_ant, n_bs_ant, n_subcarriers = csi_data.shape

    # Step 1: Compute Frobenius norm for each sample (20000 samples)
    fro_norms = np.sqrt(np.sum(np.abs(csi_data) ** 2, axis=(1, 2, 3)))  # Shape: (20000,)

    # Step 2: Normalize each sample
    # Broadcasting fro_norms to match csi_data shape for division
    csi_data = csi_data / fro_norms[:, np.newaxis, np.newaxis, np.newaxis]

    del fro_norms

    # Step 3: Scale to desired factor
    scaling_factor = np.sqrt(n_ue_ant * n_bs_ant)
    csi_data = csi_data * scaling_factor

    beamspace_magnitudes = []
    batch_size = 1000
    # Process in batches to reduce memory usage
    for start_idx in tqdm(range(0, csi_data.shape[0], batch_size)):
        end_idx = min(start_idx + batch_size, csi_data.shape[0])

        batch = csi_data[start_idx:end_idx]

        # Step 1: Apply 2D Fourier Transform across UE and BS antennas
        beamspace_batch = np.fft.fft2(batch, axes=(1, 2))

        # Step 2: Normalize Beamspace Data (Optional)
        beamspace_batch /= np.sqrt(batch.shape[1] * batch.shape[2])  # Normalize by number of antennas

        test = np.abs(beamspace_batch)
        print(test.shape)
        # Store the results in the output array
        beamspace_magnitudes.append(test)

    beamspace_magnitudes = np.concatenate(beamspace_magnitudes, axis=0)

    return beamspace_magnitudes.reshape(n_samples, -1)



def generate_augmented_data(X, valid_anchors, valid_positions):
    """
    Generate augmented data for training the Siamese Network.
    Args:
        X: Input features (numpy array)
        valid_anchors: List of indices for valid anchors
        valid_positions: List of positions corresponding to valid anchors
    Returns:
        Augmented data (numpy array)
    """
    np.random.seed(42)

    X_aug = []
    valid_anchors_aug = []
    valid_positions_aug = []
    for i,idx in enumerate(valid_anchors):

        for _ in range(3):
            x1 = X[idx]

            N = 0.5 * (np.random.randn(x1.shape[0]))
            x1 += N

            # print(f"X1: {x1.shape}, X: {X.shape}")
            X_aug.append(x1)
            valid_anchors_aug.append(len(X) + i)
            valid_positions_aug.append(valid_positions[i])
            

    print("DONE!")
    X = np.concatenate((X, np.array(X_aug)))
    valid_anchors = np.concatenate((valid_anchors, np.array(valid_anchors_aug))) 
    valid_positions = np.concatenate((valid_positions, np.array(valid_positions_aug)))

    return X, valid_anchors, valid_positions

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
    method: str = "Siamese",
    PathRaw="",
    Prefix="",
    na=1,
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
        method: Method which we want to use for generating results
        PathRaw: Path to file we want to use
        Prefix: Prefix choosing which dataset we want to use
        na: Number of file (in our case 1-3)

    Returns:
        Predicted positions array of shape (tol_samp_num, 2)
    """
    # Create result array
    loc_result = np.zeros([tol_samp_num, 2], "float")

    feature_file = PathRaw + "/" + Prefix + "FeaturesBetter" + f"{na}" + ".npy"

    X = []
    my_file = Path(feature_file)

    if my_file.exists():
        print(f"Retrieving features from {feature_file}")
        X = np.load(feature_file)
    else:
        print("Extracting features from channel data...")
        X = extract_features(H)
        print(f"Saving features to file {feature_file}")
        np.save(feature_file, X)

    # Prepare training data from anchor points that are within our current slice
    valid_anchors = []
    valid_positions = []

    for anchor in anch_pos:
        # maks = np.max(anchor[1:])
        # minn = np.min(anchor[1:])
        # print("Max: ", maks, " Min: ", minn)
        # exit(0)
        idx = int(anchor[0]) - 1  # Convert to 0-based index
        if idx < len(H):  # Only use anchors that are in our current slice
            valid_anchors.append(idx)
            valid_positions.append(anchor[1:])

    if len(valid_anchors) > 0:
        print(f"Training model with {len(valid_anchors)} anchor points...")
        X_train = X[valid_anchors]
        y_train = np.array(valid_positions)

        if method == "Siamese":
            # Initialize Siamese Network
            input_dim = X.shape[1]
            learning_rate = 0.0003
            num_epochs = 200
            device = "cuda" if torch.cuda.is_available() else "cpu"

            X_augmented, valid_anchors_augmented, y_train_augmented = generate_augmented_data(X, valid_anchors, y_train)
            
            print(f"X_augmented: {X_augmented.shape}, valid_anchors_augmented: {valid_anchors_augmented.shape}, y_train_augmented: {y_train_augmented.shape}")

            dataset = SiameseDataset(X_augmented, valid_anchors_augmented, y_train_augmented, device=device)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            model = SiameseNetwork(input_dim)
            model.to(device)
            criterion = SiameseLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Learning loop
            for epoch in (pbar := tqdm(range(num_epochs))):
                model.train()
                total_loss = 0
                for x1, x2, real_y1 in dataloader:
                    optimizer.zero_grad()

                    z1, z2 = model(x1, x2)
                    loss = criterion(x1, x2, z1, z2, real_y1)
                    loss.backward()

                    optimizer.step()
                    total_loss += loss.item()
                pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

            # Calculate predictions
            first = True
            with torch.no_grad():
                predictions = []
                for x1 in tqdm(X):
                    x1_tensor = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)
                    z1, _ = model(x1_tensor.to(device), x1_tensor.to(device))
                    predictions.append(z1.cpu().numpy())

                    if first:
                        print(z1)
                        first = False

                predictions = np.array(predictions)
            torch.save(model.state_dict(), "siamese_model_state_dict.pth")
        elif method == "XGBoost":
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            predictions = xgb_model.predict(X)
        else:
            raise ValueError(f"Invalid method: {method}")

        # Fill the corresponding positions in the result array
        for i in range(len(H)):
            loc_result[i] = predictions[i]

    return loc_result


def plot_distance_distribution(prediction_file: str, ground_truth_file: str, save_path: str = None):
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


def plot_scatter_GroundTruth(ground_truth_file0, ground_truth_file1, ground_truth_file2):
    """
    Plot scatter points
    """
    ground_truth0 = np.loadtxt(ground_truth_file0)
    ground_truth1 = np.loadtxt(ground_truth_file1)
    ground_truth2 = np.loadtxt(ground_truth_file2)

    ground_truth = np.concatenate((ground_truth0, ground_truth1, ground_truth2), axis=0)
    plt.figure(figsize=(10, 6))
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label="Ground Truth")
    plt.show()


def plot_predictions_vs_truth(predictions, ground_truth):
    """
    Plot scatter points for predicted and ground truth positions
    """
    predictions = np.loadtxt(predictions)
    ground_truth = np.loadtxt(ground_truth)

    predictions = predictions[:2]
    ground_truth = ground_truth[:2]
    plt.figure(figsize=(10, 6))
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label="Ground Truth")
    plt.scatter(predictions[:, 0], predictions[:, 1], label="Predictions")
    plt.legend()
    plt.show()


def evaluate_score(prediction_file: str, ground_truth_file: str, dataset_ind: str) -> float:
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
