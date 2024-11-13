import os
import time
from pathlib import Path

import numpy as np

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

    Ridx = 0  # Flag defining the round of the competition, used for define where to read data。0:Test; 1: 1st round; 2: 2nd round ...
    PathRaw = PathSet[Ridx]
    Prefix = PrefixSet[Ridx]

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
        csi_path = PathRaw + "/" + Prefix + "InputData" + na + ".txt"
        csi_file = PathRaw + "/" + Prefix + "InputData" + na + ".npy"

        my_file = Path(csi_file)

        if my_file.exists():
            H = np.load(
                csi_file
            )  # if saved in npy, you can load npy file instead of txt
        else:
            print(slice_num)
            H = []
            for slice_idx in range(
                slice_num
            ):  # range(slice_num): # Read in channel data in a loop. In each loop, only one slice of channel is read in
                print("Loading input CSI data of slice " + str(slice_idx))
                slice_lines = read_slice_of_file(
                    csi_path,
                    slice_idx * slice_samp_num,
                    (slice_idx + 1) * slice_samp_num,
                )
                Htmp = np.loadtxt(slice_lines)
                Htmp = np.reshape(Htmp, (slice_samp_num, 2, sc_num, ant_num, port_num))
                Htmp = Htmp[:, 0, :, :, :] + 1j * Htmp[:, 1, :, :, :]
                Htmp = np.transpose(Htmp, (0, 3, 2, 1))
                if np.size(H) == 0:
                    H = Htmp
                else:
                    H = np.concatenate((H, Htmp), axis=0)
            H = H.astype(np.complex64)  # trunc to complex64 to reduce storage

            np.save(
                csi_file, H
            )  # After reading the file, you may save txt file into npy, which is faster for python to read

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
        with open(PathRaw + "/" + Prefix + "Output" + na + ".txt", "w") as f:
            np.savetxt(f, result, fmt="%.4f %.4f")

        # This help to evaluate the running time, can be removed!
        tEnd = time.perf_counter()
        print("Total time consuming = {}s\n\n".format(round(tEnd - tStart, 3)))

        output_path = os.path.join(PathRaw, f"Dataset{Ridx}Output{na}.txt")
        ground_truth_path = os.path.join(PathRaw, f"Dataset0GroundTruth{na}.txt")
        if os.path.exists(ground_truth_path):
            print("Evaluating results...")
            score = evaluate_score(output_path, ground_truth_path, int(na))

            plot_path = os.path.join(PathRaw, f"Dataset{Ridx}ErrorDist{na}.png")
            plot_distance_distribution(
                output_path, ground_truth_path, save_path=plot_path
            )
            print(f"\nVisualization saved to: {plot_path}")
