import os
import time

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


Current_Best_Sum_Score = [1066763.46, 2213384.89, 2283934.90]
Current_Best_Mean_Score = [53.34, 110.67, 114.20]


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
                    # np.median(H_mag[i]),  # Overall median
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
            n_neighbors=min(10, len(valid_anchors)), weights="distance", metric="cosine"
        )
        knn.fit(X_train, y_train)

        # Predict positions for the current slice
        predictions = knn.predict(X_scaled)

        # Fill the corresponding positions in the result array
        for i in range(len(H)):
            loc_result[i] = predictions[i]

    return loc_result


# Read in the configuration file
def read_cfg_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
        line_fmt = [line.rstrip("\n").split(" ") for line in lines]
    info = line_fmt
    bs_pos = list([float(info[0][0]), float(info[0][1]), float(info[0][2])])
    tol_samp_num = int(info[1][0])
    anch_samp_num = int(info[2][0])
    port_num = int(info[3][0])
    ant_num = int(info[4][0])
    sc_num = int(info[5][0])
    return bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num


# Read in the info related to the anchor points
def read_anch_file(file_path, anch_samp_num):
    anch_pos = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        line_fmt = [line.rstrip("\n").split(" ") for line in lines]
    for line in line_fmt:
        tmp = np.array([int(line[0]), float(line[1]), float(line[2])])
        if np.size(anch_pos) == 0:
            anch_pos = tmp
        else:
            anch_pos = np.vstack((anch_pos, tmp))
    return anch_pos


# The channel file is large, read in channels in smaller slices
def read_slice_of_file(file_path, start, end):
    with open(file_path, "r") as file:
        slice_lines = list(itertools.islice(file, start, end))
    return slice_lines


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
    print(f"Total Score (sum of distances): {Current_Best_Sum_Score[dataset_ind]:.2f} meters")
    print(f"Mean distance per point: {Current_Best_Mean_Score[dataset_ind]:.2f} meters")
    print(f"Number of points evaluated: {len(distances)}")
    print("========================")

    print(f"\n=== Evaluation Results ===")
    print(f"Total Score (sum of distances): {total_score:.2f} meters")
    print(f"Mean distance per point: {mean_distance:.2f} meters")
    print(f"Number of points evaluated: {len(distances)}")
    print("========================")
    

    return total_score


from input_handling import read_anch_file, read_cfg_file, read_slice_of_file
from utils import calcLoc, evaluate_score, plot_distance_distribution

if __name__ == "__main__":
    print("<<< Welcome to 2024 Wireless Algorithm Contest! >>>\n")
    ## For ease of data managenment, input data for different rounds are stored in different folders. Feel free to define your own
    PathSet = {
        0: "../dataset0",
        1: "../CompetitionData1",
        2: "../CompetitionData2",
        3: "../CompetitionData3",
    }
    PrefixSet = {0: "Dataset0", 1: "Dataset1", 2: "Dataset2", 3: "Round3"}
    PrefixDataSet = {0: "Round0", 1: "Dataset1", 2: "Round2", 3: "Round3"}

    Ridx = 0  # Flag defining the round of the competition, used for define where to read data。0:Test; 1: 1st round; 2: 2nd round ...
    PathRaw = PathSet[Ridx]
    Prefix = PrefixSet[Ridx]
    PrefixData = PrefixDataSet[Ridx]

    ### Get all files in the folder related to the competition. Data for other rounds should be kept in a different folder
    files = os.listdir(PathRaw)

    names = []
    for f in sorted(files):
        if f.find("CfgData") != -1 and f.endswith(".txt"):
            names.append(f.split("CfgData")[-1].split(".txt")[0])

    for na in names:
        FileIdx = int(na)
        print("Processing Round " + str(Ridx) + " Case " + str(na))

        # Read in the configureation file: RoundYCfgDataX.txt
        print("Loading configuration data file")
        cfg_path = PathRaw + "/" + Prefix + "CfgData" + na + ".txt"
        bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num = read_cfg_file(
            cfg_path
        )

        # Read in info related to the anchor points: RoundYInputPosX.txt
        print("Loading input position file")
        anch_pos_path = PathRaw + "/" + Prefix + "InputPos" + na + ".txt"
        anch_pos = read_anch_file(anch_pos_path, anch_samp_num)

        # Read in channel data:  RoundYInputDataX.txt
        slice_samp_num = 1000  # number of samples in each slice
        slice_num = int(tol_samp_num / slice_samp_num)  # total number of slices
        csi_path = PathRaw + "/" + PrefixData + "InputData" + na + ".txt"

        # print(slice_num)
        # H = []
        # for slice_idx in range(
        #     slice_num
        # ):  # range(slice_num): # Read in channel data in a loop. In each loop, only one slice of channel is read in
        #     print("Loading input CSI data of slice " + str(slice_idx))
        #     slice_lines = read_slice_of_file(
        #         csi_path, slice_idx * slice_samp_num, (slice_idx + 1) * slice_samp_num
        #     )
        #     Htmp = np.loadtxt(slice_lines)
        #     Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
        #     Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
        #     Htmp = np.transpose(Htmp, (0, 3, 2, 1))
        #     if np.size(H) == 0:
        #         H = Htmp
        #     else:
        #         H = np.concatenate((H, Htmp), axis=0)
        # H = H.astype(np.complex64)  # trunc to complex64 to reduce storage

        csi_file = PathRaw + "/" + Prefix + "InputData" + na + ".npy"
        # np.save(
        #     csi_file, H
        # )  # After reading the file, you may save txt file into npy, which is faster for python to read
        H = np.load(csi_file) # if saved in npy, you can load npy file instead of txt

        tStart = time.perf_counter()

        print("Calculating localization results")
        result = calcLoc(
            H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num
        )  # This function should be implemented by yourself

        # Replace the position information for anchor points with ground true coordinates
        for idx in range(anch_samp_num):
            rowIdx = int(anch_pos[idx][0] - 1)
            result[rowIdx] = np.array([anch_pos[idx][1], anch_pos[idx][2]])

        # Output, be careful with the precision
        print("Writing output position file")
        with open(PathRaw + "/" + PrefixData + "Output" + na + ".txt", "w") as f:
            np.savetxt(f, result, fmt="%.4f %.4f")

        # This help to evaluate the running time, can be removed!
        tEnd = time.perf_counter()
        print("Total time consuming = {}s\n\n".format(round(tEnd - tStart, 3)))

        output_path = os.path.join(PathRaw, f"Round{Ridx}OutputPos{na}.txt")
        ground_truth_path = os.path.join(PathRaw, f"Dataset0GroundTruth{na}.txt")
        if os.path.exists(ground_truth_path):
            print("Evaluating results...")
            score = evaluate_score(output_path, ground_truth_path, int(na))

            plot_path = os.path.join(PathRaw, f"Round{Ridx}ErrorDist{na}.png")
            plot_distance_distribution(
                output_path, ground_truth_path, save_path=plot_path
            )
            print(f"\nVisualization saved to: {plot_path}")
