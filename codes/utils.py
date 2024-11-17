import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost as xgb
from kmeans import create_model, get_deep_features, train_model
from siamese import SiameseDataset, SiameseLoss, SiameseNetwork
from sklearn.manifold import MDS
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftshift
from itertools import combinations
from pathlib import Path
from sklearn.model_selection import GridSearchCV


from torch.utils.data import DataLoader
from tqdm import tqdm

Current_Best_Sum_Score = [1059861.98, 1476891.76, 1658852.85]
Current_Best_Mean_Score = [21.39, 48.17, 68.93]




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
    PathRaw="",
    Prefix="",
    na = 1
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
        return np.array(features)

    

    def extract_features_v2(csi_data, normalize=True, verbose=True):
        return features


    def extract_features_v3(csi_data, normalize=True, verbose=True):
        return features



    def extract_features_kmeans(H_data):
            return None

    # Extract features from available channel data
    print("Extracting features...")
    

    # If not third slice is used, extract additional features
    # if int(na) != 3:
    #     X_new = extract_features_v3(H) 
    
    
    feature_file = PathRaw + "/" + Prefix + "FeaturesBetter" + f"{na}" + ".npy"
    print(f"Saving/Retrieving features to/from {feature_file}")


    X = []
    my_file = Path(feature_file)

    if my_file.exists():
        print("Loading features from file...")
        X = np.load(feature_file)
    else:
        print("Extracting features from channel data...")
        X = extract_features_v4(H) if not kmeans_features else extract_features_kmeans(H)
        print("Saving features to file...")
        np.save(
            feature_file, X
        ) 

    # # If not third slice is used, extract additional features
    # if int(na) != 3:
    #     X = np.column_stack((X, X_new))

    # return
    X_scaled = X

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
            num_epochs = 1_000
            dataset = SiameseDataset(X_scaled, valid_anchors, y_train)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
            model = SiameseNetwork(input_dim)
            model.to("cuda")
            criterion = SiameseLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in (pbar := tqdm(range(num_epochs))):
                model.train()
                total_loss = 0
                for x1, x2, real_x1 in dataloader:
                    optimizer.zero_grad()

                    z1, z2, _ = model(x1, x2)
                    loss = criterion(x1, x2, z1, z2, real_x1)
                    loss.backward()

                    optimizer.step()
                    total_loss += loss.item()
                pbar.set_description(
                    f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}"
                )

            first = True
            with torch.no_grad():
                pairs = create_pairs(X_scaled)
                predictions = []
                for x1, x2 in pairs:
                    x1_tensor = torch.tensor(x1, dtype=torch.float32).unsqueeze(0)
                    x2_tensor = torch.tensor(x2, dtype=torch.float32).unsqueeze(0)

                    z1, z2, output = model(x1_tensor.to("cuda"), x2_tensor.to("cuda"))
                    predictions.append(z1.cpu().numpy())

                    if first:
                        print(z1)
                        first = False
                    # print(z1)

                predictions = np.array(predictions)
            torch.save(model.state_dict(), "siamese_model_state_dict.pth")
        elif method == "XGBoost":
            # xgb_model = xgb.XGBRegressor()
            # reg_cv = GridSearchCV(xgb_model, {"colsample_bytree":[1.0],"min_child_weight":[1.0,1.2]
            #                 ,'max_depth': [3,4,6], 'n_estimators': [50,100,200]}, verbose=2, n_jobs=-1)
            # reg_cv.fit(X_train,y_train)
            # print(reg_cv.best_params_)
            xgb_model = xgb.XGBRegressor(n_estimators = 100, max_depth = 6, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            predictions = xgb_model.predict(X_scaled)
        elif method == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_scaled)
        elif method == "MDS":
            mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42, n_jobs=-1, verbose=1)
            predictions = mds.fit_transform(X_scaled)
        elif method == "KNN":
            # Train KNN model
            knn = KNeighborsRegressor(
                n_neighbors=min(5, len(valid_anchors)), weights="distance"
            )
            knn.fit(X_train, y_train)

            # Predict positions for the current slice
            predictions = knn.predict(X_scaled)
        elif method == "skowi_test":
            knn = KNeighborsRegressor(
                n_neighbors=min(5, len(valid_anchors)), weights="uniform"
            )
            knn.fit(X_train, y_train)

            neigh = NearestNeighbors(n_neighbors=5)
            neigh.fit(X_train)
            
            res = neigh.kneighbors([X_scaled[0]])
            print("O WOW ")
            print(res)
            res = res[1]
            print(y_train[res[0]])

            plt.figure(figsize=(10, 6))
            plt.scatter(y_train[res[0], 0], y_train[res[0], 1], label="K-Neigherst Neighbors")

            predictions = np.zeros((len(X_scaled), 2))

            predictions = knn.predict(X_scaled)

            ground_truth = np.loadtxt("../dataset0/Dataset0GroundTruth1.txt")
            plt.scatter(ground_truth[0, 0], ground_truth[0, 1], label="Ground Truth")

            plt.scatter(predictions[0, 0], predictions[0, 1], label="Predictions")

            plt.legend()
            plt.show()

            # plt.figure(figsize=(10, 6))
            # plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label="Ground Truth")

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
    predictions =   np.loadtxt(predictions)
    ground_truth = np.loadtxt(ground_truth)

    predictions = predictions[:2]
    ground_truth = ground_truth[:2]
    plt.figure(figsize=(10, 6))
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label="Ground Truth")
    plt.scatter(predictions[:, 0], predictions[:, 1], label="Predictions")
    plt.legend()
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



def extract_features_v4(csi_data, normalize=True, verbose=True):

    n_samples, n_ue_ant, n_bs_ant, n_subcarriers = csi_data.shape

    # Step 1: Compute Frobenius norm for each sample (20000 samples)
    fro_norms = np.sqrt(np.sum(np.abs(csi_data)**2, axis=(1, 2, 3)))  # Shape: (20000,)


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
    
def process_csi_data(csi_data):
    """
    Transform CSI data from frequency to delay domain, take absolute values, and normalize.
    
    Parameters:
    -----------
    csi_data : numpy.ndarray
        CSI data of shape (n_samples, n_ue_antennas, n_bs_antennas, n_subcarriers)
    
    Returns:
    --------
    numpy.ndarray
        Processed CSI data with the same shape
    """
    # Get the shape of input data
    n_samples, n_ue_ant, n_bs_ant, n_subcarriers = csi_data.shape
    
    # Initialize output array
    processed_data = np.zeros_like(csi_data, dtype=np.float32)

    print("Processing CSI data...")
    
    # Process each sample
    for i in tqdm(range(n_samples)):
        for j in range(n_ue_ant):
            for k in range(n_bs_ant):
                # Get frequency domain data for current antenna pair
                freq_data = csi_data[i, j, k, :]
                
                # Transform to delay domain using IFFT
                delay_data = np.fft.ifft(freq_data)
                
                # Calculate absolute values
                abs_data = np.abs(delay_data)
                
                # Normalize to unit norm
                norm = np.linalg.norm(abs_data)
                if norm > 0:  # Avoid division by zero
                    normalized_data = abs_data / norm
                else:
                    normalized_data = abs_data
                
                # Store the result
                processed_data[i, j, k, :] = normalized_data
    
    return processed_data.reshape(n_samples, -1)