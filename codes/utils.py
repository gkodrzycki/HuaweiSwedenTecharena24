import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
    loc_result = np.zeros([tol_samp_num, 2], "float")


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
        
        if verbose:
            print("\n1. Computing Basic CSI Features...")
            print("   - Converting to amplitude and phase")
        amplitude = np.abs(csi_data)
        phase = np.angle(csi_data)
        phase_unwrapped = np.unwrap(phase, axis=3)
        
        if verbose:
            print("\n2. Computing Statistical Features...")
            print("   - Processing each UE-BS antenna pair")
        
        feature_count = 0
        for ue_ant in range(n_ue_ant):
            for bs_ant in range(n_bs_ant):
                if verbose:
                    print(f"   - Processing UE ant {ue_ant+1}/{n_ue_ant}, BS ant {bs_ant+1}/{n_bs_ant}")
                
                amp_mean = np.mean(amplitude[:, ue_ant, bs_ant, :], axis=1)
                amp_std = np.std(amplitude[:, ue_ant, bs_ant, :], axis=1)
                amp_var = np.var(amplitude[:, ue_ant, bs_ant, :], axis=1)
                
                phase_mean = np.mean(phase_unwrapped[:, ue_ant, bs_ant, :], axis=1)
                phase_std = np.std(phase_unwrapped[:, ue_ant, bs_ant, :], axis=1)
                phase_var = np.var(phase_unwrapped[:, ue_ant, bs_ant, :], axis=1)
                
                features.extend([amp_mean, amp_std, amp_var, phase_mean, phase_std, phase_var])
                feature_count += 6
        
        if verbose:
            print(f"   Features extracted so far: {feature_count}")
        
        if verbose:
            print("\n3. Computing Cross-antenna Features...")
        
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
        
        # # 5. Include original features
        # if verbose:
        #     print("\n5. Adding original CSI features...")
        
        # original_features = extract_features_v2(csi_data, normalize=False, verbose=False)
        # features.append(original_features)
        # feature_count += original_features.shape[1]
        
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

    print("Extracting features...")
    
    if int(na) != 3:
        X_new = extract_features_v3(H) 
    
    
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

    if int(na) != 3:
        X = np.column_stack((X, X_new))

    X_scaled = X

    valid_anchors = []
    valid_positions = []

    for anchor in anch_pos:
        idx = int(anchor[0]) - 1  
        if idx < len(H): 
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
            xgb_model = xgb.XGBRegressor(n_estimators = 100, max_depth = 7, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            predictions = xgb_model.predict(X_scaled)
        elif method == "RandomForest":
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=42)
            rf.fit(X_train, y_train)
            predictions = rf.predict(X_scaled)
        elif method == "MDS":
            mds = MDS(n_components=2, dissimilarity="euclidean", random_state=42, n_jobs=-1)
            predictions = mds.fit_transform(X_scaled)
        elif method == "KNN":
            knn = KNeighborsRegressor(
                n_neighbors=min(5, len(valid_anchors)), weights="distance"
            )
            knn.fit(X_train, y_train)

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

        else:
            raise ValueError(f"Invalid method: {method}")

        for i in range(len(H)):
            loc_result[i] = predictions[i]

    return loc_result


def enhance_csi_features(csi_data, dh=0.5, dv=2.0, fc=3.5e9, subcarrier_spacing=240e3, verbose=True):
    """
    Extract enhanced CSI features considering array geometry and signal parameters
    
    Parameters:
    -----------
    csi_data : ndarray
        Complex CSI data of shape (n_samples, n_ue_ant, n_bs_ant, n_subcarriers)
    dh : float
        Horizontal antenna spacing in wavelengths (default=0.5)
    dv : float
        Vertical antenna spacing in wavelengths (default=2.0)
    fc : float
        Carrier frequency in Hz (default=3.5GHz)
    subcarrier_spacing : float
        Effective subcarrier spacing in Hz (default=240kHz)
    """
    
    n_samples, n_ue_ant, n_bs_ant, n_subcarriers = csi_data.shape
    features = []
    
    # Array configuration
    N_POL = 2
    N_COLS = 8
    N_ROWS = 4
    
    # Wavelength calculation
    c = 3e8  # speed of light
    wavelength = c / fc
    
    # Convert to physical distances
    dx = dh * wavelength
    dy = dv * wavelength
    
    # Reshape data to separate polarizations and spatial structure
    csi_pol = csi_data.reshape(n_samples, n_ue_ant, N_POL, N_ROWS, N_COLS, n_subcarriers)
    
    if verbose:
        print("\n1. Computing Spatial-Frequency Features...")
    
    for pol in range(N_POL):
        # 1. Beam-Space Domain Transform
        beam_space = np.fft.fft2(csi_pol[:, :, pol, :, :, :], axes=(3, 4))
        beam_mag = np.abs(beam_space)
        
        # Find indices of dominant beams
        beam_mag_flat = beam_mag.reshape(*beam_mag.shape[:3], -1)
        flat_max_idx = np.argmax(beam_mag_flat, axis=-1)
        spatial_dims = beam_mag.shape[3:5]  # Dynamically get (N_ROWS, N_COLS)
        max_beam_idx = np.unravel_index(flat_max_idx, spatial_dims)

        # Compute beam angles
        row_idx, col_idx = max_beam_idx
        normalized_row = (row_idx - (spatial_dims[0] // 2)) / (spatial_dims[0] // 2)
        normalized_col = (col_idx - (spatial_dims[1] // 2)) / (spatial_dims[1] // 2)

        magnitude = np.sqrt(normalized_row**2 + normalized_col**2)
        magnitude = np.clip(magnitude, -1, 1)  # Ensure valid input for arcsin
        beam_angles = np.arcsin(magnitude)

        features.extend([
            np.mean(beam_angles, axis=(1, 2)),
            np.std(beam_angles, axis=(1, 2))
        ])
        
        # 2. Delay-Angle Domain Features
        delay_angle = np.fft.fft(csi_pol[:, :, pol, :, :, :], axis=-1)
        delay_mag = np.abs(delay_angle)
        
        delay_spread = np.std(delay_mag, axis=-1)
        angle_spread = np.std(delay_mag, axis=(3, 4))
        
        features.extend([
            np.mean(delay_spread, axis=(1, 2)),
            np.mean(angle_spread, axis=1)
        ])
        
        # 3. Cross-Polarization Ratio (XPR)
        if pol == 0:
            xpr = np.abs(csi_pol[:, :, 0, :, :, :]) / (np.abs(csi_pol[:, :, 1, :, :, :]) + 1e-10)
            features.extend([
                np.mean(xpr, axis=(1, 2, 3, 4)),  # Correct axis specification
                np.std(xpr, axis=(1, 2, 3, 4))
            ])
    
    # 4. Subarray Features
    if verbose:
        print("2. Computing Subarray Features...")
    
    # Split array into subarrays
    for pol in range(N_POL):
        for i in range(0, N_ROWS-1, 2):
            for j in range(0, N_COLS-1, 2):
                subarray = csi_pol[:, :, pol, i:i+2, j:j+2, :]
                
                # Compute subarray correlation matrix
                corr_matrix = np.zeros((n_samples, 4, 4))
                for s in range(n_samples):
                    flat_subarray = subarray[s].reshape(-1, n_subcarriers)
                    corr_matrix[s] = np.corrcoef(flat_subarray)
                
                # Extract eigenvalues of correlation matrix
                eigenvals = np.linalg.eigvalsh(corr_matrix)
                features.extend([
                    np.mean(eigenvals, axis=1),
                    np.std(eigenvals, axis=1)
                ])
    
    # 5. Frequency-Domain Features
    if verbose:
        print("3. Computing Enhanced Frequency Features...")
    
    # Compute frequency response correlation
    freq_corr = np.zeros((n_samples, n_subcarriers-1))
    for i in range(n_subcarriers-1):
        freq_corr[:, i] = np.abs(np.sum(
            csi_data[:, :, :, i] * np.conj(csi_data[:, :, :, i+1]),
            axis=(1, 2)
        ))
    
    features.extend([
        np.mean(freq_corr, axis=1),
        np.std(freq_corr, axis=1),
        np.max(freq_corr, axis=1)
    ])
    
    # Combine all features
    features = np.column_stack(features)
    
    if verbose:
        print(f"\nTotal features extracted: {features.shape[1]}")
    
    return features

def create_enhanced_model(X_train, y_train, method="Ensemble"):
    """
    Create an enhanced model for localization
    """
    if method == "Ensemble":
        # Create base models
        models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1
            ),
            'rf': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                n_jobs=-1
            ),
            'knn': KNeighborsRegressor(
                n_neighbors=5,
                weights='distance',
                metric='manhattan'
            )
        }
        
        # Train base models
        predictions = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions[name] = model.predict(X_train)
        
        # Create meta-features
        meta_features = np.column_stack([pred for pred in predictions.values()])
        
        # Train meta-model
        meta_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.01
        )
        meta_model.fit(meta_features, y_train)
        
        return models, meta_model
    else:
        return None

def predict_location(models, meta_model, X_test):
    """
    Make predictions using the ensemble model
    """
    # Get predictions from base models
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X_test)
    
    # Create meta-features
    meta_features = np.column_stack([pred for pred in predictions.values()])
    
    # Make final predictions
    final_predictions = meta_model.predict(meta_features)
    
    return final_predictions

def enhanced_calcLoc(H, anch_pos, bs_pos, tol_samp_num, anch_samp_num, port_num, ant_num, sc_num,
                    method="Ensemble", PathRaw="", Prefix=""):
    """
    Enhanced implementation of channel-based localization
    """
    print("Extracting enhanced features...")
    
    # Extract enhanced features
    X = enhance_csi_features(H)
    
    # Save features
    feature_file = PathRaw + "/" + Prefix + "EnhancedFeatures.npy"
    np.save(feature_file, X)
    
    # Prepare training data
    valid_anchors = []
    valid_positions = []
    for anchor in anch_pos:
        idx = int(anchor[0]) - 1
        if idx < len(H):
            valid_anchors.append(idx)
            valid_positions.append(anchor[1:])
    
    if len(valid_anchors) > 0:
        X_train = X[valid_anchors]
        y_train = np.array(valid_positions)
        
        # Train enhanced model
        models, meta_model = create_enhanced_model(X_train, y_train, method)
        
        # Make predictions
        predictions = predict_location(models, meta_model, X)
        
        # Return results
        loc_result = np.zeros([tol_samp_num, 2], "float")
        for i in range(len(H)):
            loc_result[i] = predictions[i]
        
        return loc_result
    
    return np.zeros([tol_samp_num, 2], "float")


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
