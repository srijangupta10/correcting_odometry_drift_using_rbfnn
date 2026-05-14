import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR = "results_mouse_lidar"
OUTPUT_DIR = "validation_results_v2"
MIN_RUN_DIST = 100.0

CPI_LX, CPI_LY = 314.5, 312.2
CPI_RX, CPI_RY = 315.1, 313.8
N_OFFSET = 70.0
TRACK_WIDTH = N_OFFSET * 2.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

class NumpyRBFNN:
    def __init__(self, model_dir):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Missing model directory: {model_dir}")
        self.centers = np.load(os.path.join(model_dir, "centers.npy"))
        self.sigma = np.load(os.path.join(model_dir, "sigma.npy"))
        self.weights = np.load(os.path.join(model_dir, "weights.npy"))
        self.bias = np.load(os.path.join(model_dir, "bias.npy"))

    def predict(self, dx, dy):
        x = np.array([dx, dy])
        dists = np.linalg.norm(self.centers - x, axis=1)
        phi = np.exp(-(dists**2) / (2 * self.sigma**2))
        return np.dot(phi, self.weights.T) + self.bias

def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def validate_all_trajectories():
    print("Loading NumPy Edge-AI Models...")
    rbf_left = NumpyRBFNN("model_weights_left")
    rbf_right = NumpyRBFNN("model_weights_right")

    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    print(f"Found {len(csv_files)} total raw files.")

    results_summary = []

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        df = pd.read_csv(file_path)

        valid_indices = df[df['lidar_valid'] == 1].index.tolist()
        if len(valid_indices) < 2: continue

        start_pose = df.loc[valid_indices[0]]
        end_pose = df.loc[valid_indices[-1]]

        total_dist_file = np.sqrt((end_pose['lidar_x_mm'] - start_pose['lidar_x_mm'])**2 +
                                  (end_pose['lidar_y_mm'] - start_pose['lidar_y_mm'])**2)
        if total_dist_file < MIN_RUN_DIST: continue

        pose_raw = np.array([0.0, 0.0, 0.0])
        pose_nn = np.array([0.0, 0.0, 0.0])
        pose_lidar = np.array([0.0, 0.0, 0.0])

        path_raw, path_nn, path_lidar = [pose_raw.copy()], [pose_nn.copy()], [pose_lidar.copy()]

        df_mouse = df.loc[valid_indices[0]:valid_indices[-1]].copy()
        df_mouse = df_mouse[df_mouse['mouse_dx_L'].notna()].copy()
        df_lidar = df.loc[valid_indices].copy()

        start_x_lidar = df_lidar.iloc[0]['lidar_x_mm']
        start_y_lidar = df_lidar.iloc[0]['lidar_y_mm']
        start_th_lidar = df_lidar.iloc[0]['lidar_theta_rad']

        lidar_times = df_lidar['timestamp_sec'].values
        df_mouse['lidar_idx'] = np.digitize(df_mouse['timestamp_sec'], lidar_times)
        mouse_groups = df_mouse.groupby('lidar_idx')

        for i in range(1, len(df_lidar)):
            l_row = df_lidar.iloc[i]

            dx_glob = l_row['lidar_x_mm'] - start_x_lidar
            dy_glob = l_row['lidar_y_mm'] - start_y_lidar

            pose_lidar[0] = dx_glob * np.cos(-start_th_lidar) - dy_glob * np.sin(-start_th_lidar)
            pose_lidar[1] = dx_glob * np.sin(-start_th_lidar) + dy_glob * np.cos(-start_th_lidar)
            pose_lidar[2] = normalize_angle(l_row['lidar_theta_rad'] - start_th_lidar)

            path_lidar.append(pose_lidar.copy())

            if i not in mouse_groups.groups: continue
            m_interval = mouse_groups.get_group(i)

            for _, m in m_interval.iterrows():
                nom_x_l = m['mouse_dx_L'] / CPI_LX
                nom_y_l = -m['mouse_dy_L'] / CPI_LY
                nom_x_r = m['mouse_dx_R'] / CPI_RX
                nom_y_r = -m['mouse_dy_R'] / CPI_RY

                dy_raw_c = (nom_y_l + nom_y_r) / 2.0
                dx_raw_c = (nom_x_l + nom_x_r) / 2.0
                dth_raw = (nom_y_r - nom_y_l) / TRACK_WIDTH

                pose_raw[0] += dx_raw_c * np.cos(pose_raw[2]) - dy_raw_c * np.sin(pose_raw[2])
                pose_raw[1] += dx_raw_c * np.sin(pose_raw[2]) + dy_raw_c * np.cos(pose_raw[2])
                pose_raw[2] = normalize_angle(pose_raw[2] + dth_raw)
                path_raw.append(pose_raw.copy())

                err_x_l, err_y_l = 0.0, 0.0
                err_x_r, err_y_r = 0.0, 0.0

                if np.sqrt(nom_x_l**2 + nom_y_l**2) > 1e-5:
                    err_x_l, err_y_l = rbf_left.predict(nom_x_l, nom_y_l)
                if np.sqrt(nom_x_r**2 + nom_y_r**2) > 1e-5:
                    err_x_r, err_y_r = rbf_right.predict(nom_x_r, nom_y_r)

                corr_x_l, corr_y_l = nom_x_l + err_x_l, nom_y_l + err_y_l
                corr_x_r, corr_y_r = nom_x_r + err_x_r, nom_y_r + err_y_r

                dy_nn_c = (corr_y_l + corr_y_r) / 2.0
                dx_nn_c = (corr_x_l + corr_x_r) / 2.0
                dth_nn = (corr_y_r - corr_y_l) / TRACK_WIDTH

                pose_nn[0] += dx_nn_c * np.cos(pose_nn[2]) - dy_nn_c * np.sin(pose_nn[2])
                pose_nn[1] += dx_nn_c * np.sin(pose_nn[2]) + dy_nn_c * np.cos(pose_nn[2])
                pose_nn[2] = normalize_angle(pose_nn[2] + dth_nn)
                path_nn.append(pose_nn.copy())

        raw_error_mm = np.sqrt((pose_lidar[0] - pose_raw[0])**2 + (pose_lidar[1] - pose_raw[1])**2)
        nn_error_mm = np.sqrt((pose_lidar[0] - pose_nn[0])**2 + (pose_lidar[1] - pose_nn[1])**2)

        raw_err_deg = np.abs(np.degrees(normalize_angle(pose_lidar[2] - pose_raw[2])))
        nn_err_deg = np.abs(np.degrees(normalize_angle(pose_lidar[2] - pose_nn[2])))

        if not np.isnan(raw_error_mm) and not np.isnan(nn_error_mm):
            results_summary.append({
                'file': file_name,
                'dist_traveled': total_dist_file,
                'raw_err': raw_error_mm,
                'nn_err': nn_error_mm,
                'raw_err_deg': raw_err_deg,
                'nn_err_deg': nn_err_deg
            })

        pr, pnn, pl = np.array(path_raw), np.array(path_nn), np.array(path_lidar)
        plt.figure(figsize=(10, 8))
        plt.plot(pl[:, 0], pl[:, 1], 'k-', label='LiDAR Truth', linewidth=3, alpha=0.6)
        plt.plot(pr[:, 0], pr[:, 1], 'r--', label=f'Raw (Err: {raw_error_mm:.1f}mm, {raw_err_deg:.1f}°)', linewidth=2)
        plt.plot(pnn[:, 0], pnn[:, 1], 'g-', label=f'RBFNN (Err: {nn_error_mm:.1f}mm, {nn_err_deg:.1f}°)', linewidth=2)
        plt.title(f'Hybrid RBFNN Validation\n{file_name}\nDist: {total_dist_file:.1f}mm')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')

        plot_name = os.path.splitext(file_name)[0] + "_val.png"
        plt.savefig(os.path.join(OUTPUT_DIR, plot_name), dpi=150)
        plt.close()

    df_res = pd.DataFrame(results_summary)

    if len(df_res) > 0:
        df_res['raw_err'] = df_res['raw_err'].replace(0, 0.0001)
        df_res['raw_err_deg'] = df_res['raw_err_deg'].replace(0, 0.0001)

        df_res['reduction_pct'] = (1.0 - (df_res['nn_err'] / df_res['raw_err'])) * 100
        df_res['reduction_deg_pct'] = (1.0 - (df_res['nn_err_deg'] / df_res['raw_err_deg'])) * 100

        csv_path = os.path.join(OUTPUT_DIR, "validation_summary.csv")
        df_res.to_csv(csv_path, index=False)

        avg_raw_xy = df_res['raw_err'].mean()
        avg_nn_xy = df_res['nn_err'].mean()
        true_overall_reduction_xy = (1.0 - (avg_nn_xy / avg_raw_xy)) * 100

        avg_raw_deg = df_res['raw_err_deg'].mean()
        avg_nn_deg = df_res['nn_err_deg'].mean()
        true_overall_reduction_deg = (1.0 - (avg_nn_deg / avg_raw_deg)) * 100

        print("HYBRID ARCHITECTURE: FINAL VALIDATION RESULTS")

        print("POSITIONAL ERROR (X,Y)")
        print(f"   Raw Error:   {avg_raw_xy:.2f} mm")
        print(f"   RBFNN Error: {avg_nn_xy:.2f} mm")
        print(f"   Reduction:   {true_overall_reduction_xy:.2f} %")
        print("HEADING ERROR (Theta)")
        print(f"   Raw Error:   {avg_raw_deg:.2f} degrees")
        print(f"   RBFNN Error: {avg_nn_deg:.2f} degrees")
        print(f"   Reduction:   {true_overall_reduction_deg:.2f} %")
        print(f"Full table saved to: {csv_path}")
        print(f"Trajectory plots saved to: ./{OUTPUT_DIR}/")
    else:
        print("No valid trajectories processed.")

if __name__ == "__main__":
    validate_all_trajectories()