import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# This funcation calculates the positions of all channels, should be implemented by the participants
def calcLoc(
    H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num
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

    # Extract features from available channel data
    X = extract_features(H)

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
        X_train = X_scaled[valid_anchors]
        y_train = np.array(valid_positions)

        # Train KNN model
        knn = KNeighborsRegressor(
            n_neighbors=min(5, len(valid_anchors)), weights="distance"
        )
        knn.fit(X_train, y_train)

        # Predict positions for the current slice
        predictions = knn.predict(X_scaled)

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


def evaluate_score(prediction_file: str, ground_truth_file: str) -> float:
    """
    Calculate score as sum of Euclidean distances between predicted and ground truth points.

    Args:
        prediction_file: Path to the file containing predicted positions (x, y)
        ground_truth_file: Path to the file containing ground truth positions (x, y)

    Returns:
        Total score (lower is better)
    """
    predictions = np.loadtxt(prediction_file)
    ground_truth = np.loadtxt(ground_truth_file)

    distances = np.sqrt(np.sum((predictions - ground_truth) ** 2, axis=1))

    total_score = np.sum(distances)

    mean_distance = np.mean(distances)

    print(f"\n=== Evaluation Results ===")
    print(f"Total Score (sum of distances): {total_score:.2f} meters")
    print(f"Mean distance per point: {mean_distance:.2f} meters")
    print(f"Number of points evaluated: {len(distances)}")
    print("========================")

    return total_score
