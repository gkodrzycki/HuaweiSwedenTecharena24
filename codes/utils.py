import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from kmeans import create_model, get_deep_features, train_model
from sklearn.manifold import MDS
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.stats import kurtosis, skew
from scipy.fft import fft, fftshift
from itertools import combinations
from pathlib import Path



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
        
        # # 3.2 BS antenna correlations
        # if n_bs_ant > 1:
        #     if verbose:
        #         print("   - Computing BS antenna correlations")
        #     for (ant1, ant2) in combinations(range(n_bs_ant), 2):
        #         for ue_ant in range(n_ue_ant):
        #             corr_bs = np.array([np.corrcoef(amplitude[i, ue_ant, ant1, :],
        #                                         amplitude[i, ue_ant, ant2, :])[0, 1]
        #                             for i in range(n_samples)])
        #             features.append(corr_bs)
        #             feature_count += 1
        
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
        
        # # 5. Advanced Statistical Features
        # if verbose:
        #     print("\n5. Computing Advanced Statistical Features...")
        
        # for ue_ant in range(n_ue_ant):
        #     for bs_ant in range(n_bs_ant):
        #         if verbose:
        #             print(f"   - Processing UE ant {ue_ant+1}/{n_ue_ant}, BS ant {bs_ant+1}/{n_bs_ant}")
                
        #         subcarrier_mean = np.mean(amplitude[:, ue_ant, bs_ant, :], axis=0)
        #         subcarrier_std = np.std(amplitude[:, ue_ant, bs_ant, :], axis=0)
                
        #         kurt = kurtosis(amplitude[:, ue_ant, bs_ant, :], axis=1)
        #         skewness = skew(amplitude[:, ue_ant, bs_ant, :], axis=1)
                
        #         features.extend([subcarrier_mean, subcarrier_std, kurt, skewness])
        #         feature_count += 4
        
        # if verbose:
        #     print(f"   Features extracted so far: {feature_count}")
        
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
    
    
    feature_file = PathRaw + "/" + Prefix + "Features" + f"{na}" + ".npy"
    print(f"Saving/Retrieving features to/from {feature_file}")


    X = []
    my_file = Path(feature_file)

    if my_file.exists():
        print("Loading features from file...")
        X = np.load(feature_file)
    else:
        print("Extracting features from channel data...")
        X = extract_features_v2(H) if not kmeans_features else extract_features_kmeans(H)
        np.save(
            feature_file, X
        ) 

    # Normalize features
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
        if method == "XGBoost":
            xgb_model = xgb.XGBRegressor(n_estimators=10, max_depth=5)
            xgb_model.fit(X_train, y_train)
            predictions = xgb_model.predict(X_scaled)
        elif method == "MDS":
            mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42, n_jobs=-1)
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
