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

Current_Best_Mean_Score = [21.39, 48.17, 68.93]

# Extract features from channel data
def extract_features_v2(csi_data, normalize=True, verbose=True):
    """
    Extract features from CSI data with dimension order: (n_samples, n_ue_ant, n_bs_ant, n_subcarriers)
    
    Parameters:
    -----------
    csi_data : ndarray
        Complex-valued CSI measurements of shape (n_samples, n_ue_ant, n_bs_ant, n_subcarriers)
    normalize : bool, optional
        Whether to normalize features (default=True)
    verbose : bool, optional
        Whether to print progress information (default=True)
        
    Returns:
    --------
    features : ndarray
        Extracted features matrix
    """
    
    n_samples, n_ue_ant, n_bs_ant, n_subcarriers = csi_data.shape
    features = []
    
    if verbose:
        print(f"\nInput shape: {csi_data.shape}")
        print(f"Number of samples: {n_samples}")
        print(f"Number of UE antennas: {n_ue_ant}")
        print(f"Number of BS antennas: {n_bs_ant}")
        print(f"Number of subcarriers: {n_subcarriers}")
        print("\nStarting feature extraction...")
    
    # 1. Basic CSI Features
    if verbose:
        print("\n1. Computing Basic CSI Features...")
        print("   - Converting to amplitude and phase")
    amplitude = np.abs(csi_data)
    phase = np.angle(csi_data)
    phase_unwrapped = np.unwrap(phase, axis=3)
    
    # 2. Statistical Features per antenna combination
    if verbose:
        print("\n2. Computing Statistical Features...")
        print("   - Processing each UE-BS antenna pair")
    
    feature_count = 0
    for ue_ant in range(n_ue_ant):
        for bs_ant in range(n_bs_ant):
            if verbose:
                print(f"   - Processing UE ant {ue_ant+1}/{n_ue_ant}, BS ant {bs_ant+1}/{n_bs_ant}")
            
            # Amplitude statistics along subcarrier dimension
            amp_mean = np.mean(amplitude[:, ue_ant, bs_ant, :], axis=1)
            amp_std = np.std(amplitude[:, ue_ant, bs_ant, :], axis=1)
            amp_var = np.var(amplitude[:, ue_ant, bs_ant, :], axis=1)
            
            # Phase statistics along subcarrier dimension
            phase_mean = np.mean(phase_unwrapped[:, ue_ant, bs_ant, :], axis=1)
            phase_std = np.std(phase_unwrapped[:, ue_ant, bs_ant, :], axis=1)
            phase_var = np.var(phase_unwrapped[:, ue_ant, bs_ant, :], axis=1)
            
            features.extend([amp_mean, amp_std, amp_var, phase_mean, phase_std, phase_var])
            feature_count += 6
    
    if verbose:
        print(f"   Features extracted so far: {feature_count}")
    
    # 3. Cross-antenna Features
    if verbose:
        print("\n3. Computing Cross-antenna Features...")
    
    # 3.1 UE antenna correlations
    if n_ue_ant > 1:
        if verbose:
            print("   - Computing UE antenna correlations")
        for (ant1, ant2) in combinations(range(n_ue_ant), 2):
            for bs_ant in range(n_bs_ant):
                corr_ue = np.array([np.corrcoef(amplitude[i, ant1, bs_ant, :],
                                            amplitude[i, ant2, bs_ant, :])[0, 1]
                                for i in range(n_samples)])
                features.append(corr_ue)
                feature_count += 1
    
    if verbose:
        print(f"   Features extracted so far: {feature_count}")
    

    # 4. Frequency Domain Features
    if verbose:
        print("\n4. Computing Frequency Domain Features...")
    
    for ue_ant in range(n_ue_ant):
        for bs_ant in range(n_bs_ant):
            if verbose:
                print(f"   - Processing UE ant {ue_ant+1}/{n_ue_ant}, BS ant {bs_ant+1}/{n_bs_ant}")
            
            fft_data = fft(csi_data[:, ue_ant, bs_ant, :], axis=1)
            fft_mag = np.abs(fftshift(fft_data, axes=1))
            
            fft_mean = np.mean(fft_mag, axis=1)
            fft_std = np.std(fft_mag, axis=1)
            fft_peak = np.max(fft_mag, axis=1)
            fft_peak_idx = np.argmax(fft_mag, axis=1)
            
            features.extend([fft_mean, fft_std, fft_peak, fft_peak_idx])
            feature_count += 4
    
    if verbose:
        print(f"   Features extracted so far: {feature_count}")
    
    # 6. Time-Frequency Features
    if verbose:
        print("\n6. Computing Time-Frequency Features...")
    
    for ue_ant in range(n_ue_ant):
        for bs_ant in range(n_bs_ant):
            if verbose:
                print(f"   - Processing UE ant {ue_ant+1}/{n_ue_ant}, BS ant {bs_ant+1}/{n_bs_ant}")
            
            f, t, Sxx = signal.spectrogram(amplitude[:, ue_ant, bs_ant, :],
                                        fs=n_subcarriers,
                                        nperseg=min(32, n_subcarriers))
            
            spec_mean = np.mean(Sxx, axis=(1, 2))
            spec_std = np.std(Sxx, axis=(1, 2))
            spec_max = np.max(Sxx, axis=(1, 2))
            
            features.extend([spec_mean, spec_std, spec_max])
            feature_count += 3
    
    if verbose:
        print(f"   Features extracted so far: {feature_count}")
    
    # 7. Delay-Domain Features
    if verbose:
        print("\n7. Computing Delay-Domain Features...")
    
    for ue_ant in range(n_ue_ant):
        for bs_ant in range(n_bs_ant):
            if verbose:
                print(f"   - Processing UE ant {ue_ant+1}/{n_ue_ant}, BS ant {bs_ant+1}/{n_bs_ant}")
            
            delay_response = np.fft.ifft(csi_data[:, ue_ant, bs_ant, :], axis=1)
            delay_mag = np.abs(delay_response)
            
            delay_peak = np.max(delay_mag, axis=1)
            delay_mean = np.mean(delay_mag, axis=1)
            delay_spread = np.std(delay_mag, axis=1)
            
            features.extend([delay_peak, delay_mean, delay_spread])
            feature_count += 3
    
    if verbose:
        print(f"   Features extracted so far: {feature_count}")
    
    # 8. Cross-Domain Features
    if verbose:
        print("\n8. Computing Cross-Domain Features...")
    
    for ue_ant in range(n_ue_ant):
        for bs_ant in range(n_bs_ant):
            if verbose:
                print(f"   - Processing UE ant {ue_ant+1}/{n_ue_ant}, BS ant {bs_ant+1}/{n_bs_ant}")
            
            freq_domain = np.abs(fft(csi_data[:, ue_ant, bs_ant, :], axis=1))
            delay_domain = np.abs(np.fft.ifft(csi_data[:, ue_ant, bs_ant, :], axis=1))
            
            cross_corr = np.array([np.corrcoef(freq_domain[i, :], 
                                            delay_domain[i, :])[0, 1]
                                for i in range(n_samples)])
            features.append(cross_corr)
            feature_count += 1
    
    # Combine all features
    features = np.column_stack(features)
    
    if verbose:
        print(f"\nTotal number of features extracted: {features.shape[1]}")
    
    # Normalize if requested
    if normalize:
        if verbose:
            print("\nNormalizing features...")
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-10)
    
    if verbose:
        print("\nFeature extraction completed!")
        print(f"Final feature matrix shape: {features.shape}")
    
    return features

def extract_features_v3(csi_data, normalize=True, verbose=True):
    """
    Extract features from CSI data with improved spatial and polarization awareness
    
    Parameters:
    -----------
    csi_data : ndarray
        Complex-valued CSI measurements of shape (n_samples, n_ue_ant, n_bs_ant, n_subcarriers)
        where n_bs_ant=64 (32 antennas × 2 polarizations)
    normalize : bool, optional
        Whether to normalize features (default=True)
    verbose : bool, optional
        Whether to print progress information (default=True)
        
    Returns:
    --------
    features : ndarray
        Extracted features matrix
    """
    
    n_samples, n_ue_ant, n_bs_ant, n_subcarriers = csi_data.shape
    features = []
    
    # Constants for antenna array structure
    N_POL = 2  # Number of polarizations
    N_COLS = 8  # Number of columns per polarization
    N_ROWS = 4  # Number of rows per polarization
    
    if verbose:
        print(f"\nInput shape: {csi_data.shape}")
        print("\nAntenna array structure:")
        print(f"- {N_POL} polarizations")
        print(f"- {N_COLS} columns per polarization")
        print(f"- {N_ROWS} rows per polarization")
    
    # Reshape data to separate polarizations
    # Shape: (n_samples, n_ue_ant, 2, 32, n_subcarriers)
    csi_pol = csi_data.reshape(n_samples, n_ue_ant, N_POL, -1, n_subcarriers)
    
    feature_count = 0
    
    # 1. Polarization-specific features
    if verbose:
        print("\n1. Computing Polarization-specific Features...")
    
    for ue_ant in range(n_ue_ant):
        for pol in range(N_POL):
            # Basic statistics per polarization
            pol_amp = np.abs(csi_pol[:, ue_ant, pol, :, :])
            pol_phase = np.angle(csi_pol[:, ue_ant, pol, :, :])
            
            # Amplitude statistics
            features.extend([
                np.mean(pol_amp, axis=(1, 2)),  # Mean across antennas and subcarriers
                np.std(pol_amp, axis=(1, 2)),
                np.var(pol_amp, axis=(1, 2))
            ])
            
            # Phase statistics
            features.extend([
                np.mean(pol_phase, axis=(1, 2)),
                np.std(pol_phase, axis=(1, 2)),
                np.var(pol_phase, axis=(1, 2))
            ])
            
            feature_count += 6
    
    # 2. Cross-polarization features
    if verbose:
        print("\n2. Computing Cross-polarization Features...")
    
    for ue_ant in range(n_ue_ant):
        # Correlation between polarizations
        pol1_data = np.abs(csi_pol[:, ue_ant, 0, :, :])
        pol2_data = np.abs(csi_pol[:, ue_ant, 1, :, :])
        
        cross_pol_corr = np.array([
            np.corrcoef(pol1_data[i].flatten(), pol2_data[i].flatten())[0, 1]
            for i in range(n_samples)
        ])
        features.append(cross_pol_corr)
        feature_count += 1
    
    # 3. Spatial domain features (using array structure)
    if verbose:
        print("\n3. Computing Spatial Domain Features...")
    
    # Reshape to get column structure
    # Shape: (n_samples, n_ue_ant, n_pol, n_cols, n_rows, n_subcarriers)
    csi_spatial = csi_pol.reshape(n_samples, n_ue_ant, N_POL, N_COLS, N_ROWS, n_subcarriers)
    
    for ue_ant in range(n_ue_ant):
        for pol in range(N_POL):
            # Column-wise correlation
            for col in range(N_COLS-1):
                col_corr = np.array([
                    np.corrcoef(
                        np.abs(csi_spatial[i, ue_ant, pol, col]).flatten(),
                        np.abs(csi_spatial[i, ue_ant, pol, col+1]).flatten()
                    )[0, 1]
                    for i in range(n_samples)
                ])
                features.append(col_corr)
                feature_count += 1
            
            # Row-wise correlation
            for row in range(N_ROWS-1):
                row_corr = np.array([
                    np.corrcoef(
                        np.abs(csi_spatial[i, ue_ant, pol, :, row]).flatten(),
                        np.abs(csi_spatial[i, ue_ant, pol, :, row+1]).flatten()
                    )[0, 1]
                    for i in range(n_samples)
                ])
                features.append(row_corr)
                feature_count += 1
    
    # 4. Beam-space features
    if verbose:
        print("\n4. Computing Beam-space Features...")
    
    for ue_ant in range(n_ue_ant):
        for pol in range(N_POL):
            # Apply 2D FFT across spatial dimensions
            beam_space = np.fft.fft2(
                csi_spatial[:, ue_ant, pol, :, :, :],
                axes=(1, 2)  # Apply across columns and rows
            )
            beam_mag = np.abs(beam_space)
            
            # Extract beam-space statistics
            features.extend([
                np.mean(beam_mag, axis=(1, 2, 3)),  # Mean across spatial and frequency
                np.std(beam_mag, axis=(1, 2, 3)),
                np.max(beam_mag, axis=(1, 2, 3))
            ])
            feature_count += 3
    
    # Combine all features
    features = np.column_stack(features)
    
    if normalize:
        if verbose:
            print("\nNormalizing features...")
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-10)
    
    if verbose:
        print("\nFeature extraction completed!")
        print(f"Final feature matrix shape: {features.shape}")
        #print(f"New features added: {feature_count - original_features.shape[1]}")
    
    return features

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
    method: str = "XGBoost",
    PathRaw="",
    Prefix="",
    na=1
):
    """
    Enhanced implementation of channel-based localization using XGBoost with feature engineering
    """

    print("Initializing enhanced localization process...")
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error

    # Feature extraction
    def extract_features_combined(H_data):
        """
        Combines features from different extraction methods (v2, v3).
        """
        X_v2 = extract_features_v2(H_data, verbose=False)
        X_v3 = extract_features_v3(H_data, verbose=False)
        return np.column_stack((X_v2, X_v3))

    # Feature importance-based selection
    def select_important_features(X, y, top_k=50):
        model = xgb.XGBRegressor()
        model.fit(X, y)
        importance = model.feature_importances_
        top_indices = np.argsort(importance)[-top_k:]
        return X[:, top_indices]

    # Step 1: Extract features from channel data
    feature_file = PathRaw + "/" + Prefix + "Features" + f"{na}" + "_combined.npy"
    X = []
    if Path(feature_file).exists():
        print("Loading features from file...")
        X = np.load(feature_file)
    else:
        print("Extracting features from channel data...")
        X = extract_features_combined(H)
        print(f"Saving features to {feature_file}")
        np.save(feature_file, X)

    # Step 2: Prepare anchor data for training
    valid_anchors = []
    valid_positions = []

    for anchor in anch_pos:
        idx = int(anchor[0]) - 1  # Convert to 0-based index
        if idx < len(H):  # Ensure anchor index is valid
            valid_anchors.append(idx)
            valid_positions.append(anchor[1:])  # Extract x and y positions only

    if len(valid_anchors) == 0:
        raise ValueError("No valid anchors found for training. Check anchor data.")

    X_train = X[valid_anchors]
    y_train = np.array(valid_positions)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Validation step
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"Mismatch in training data sizes: X_train rows ({X_train.shape[0]}) vs y_train rows ({y_train.shape[0]})"
        )

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    print(f"Post-split shapes: X_train: {X_train.shape}, y_train: {y_train.shape}, "
        f"X_val: {X_val.shape}, y_val: {y_val.shape}")

    # Proceed with XGBoost training
    model = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )

    # Step 5: Predict positions for all samples
    print("Predicting locations for all samples...")
    X_scaled = X  # Ensure consistent normalization across training and prediction
    # X_scaled = select_important_features(X_scaled, y_train)  # Apply feature selection
    y_pred = model.predict(X_scaled)

    # Step 6: Store predictions
    loc_result = np.zeros([tol_samp_num, 2], "float")
    loc_result[:, 0] = y_pred[:, 0]
    loc_result[:, 1] = y_pred[:, 1]
    print("Localization complete.")

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
