import os
import glob
import pandas as pd
import numpy as np

INPUT_FOLDER = "results_mouse_lidar"
OUTPUT_FOLDER = "processed_data_v2"

CPI_LX, CPI_LY = 314.5, 312.2
CPI_RX, CPI_RY = 315.1, 313.8

N_OFFSET = 70.0
MIN_RUN_DIST = 100.0
MIN_STEP_DIST = 1.0

def normalize_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def process_master_dataset():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

    if not csv_files:
        print(f"No CSV files found in '{INPUT_FOLDER}'.")
        return

    print(f"Found {len(csv_files)} files. Executing Phase 1, 2 & 3...\n")

    passed_count = 0
    skipped_count = 0
    debug_printed = False

    for file_path in csv_files:
        file_name = os.path.basename(file_path)

        try:
            df = pd.read_csv(file_path)

            valid_indices = df[df['lidar_valid'] == 1].index.tolist()
            if len(valid_indices) < 2:
                skipped_count += 1
                continue

            start_pose = df.loc[valid_indices[0]]
            end_pose = df.loc[valid_indices[-1]]

            net_displacement = np.sqrt((end_pose['lidar_x_mm'] - start_pose['lidar_x_mm'])**2 +
                                       (end_pose['lidar_y_mm'] - start_pose['lidar_y_mm'])**2)

            if net_displacement <= MIN_RUN_DIST:
                skipped_count += 1
                continue

            passed_count += 1

            df_mouse = df[df['mouse_dx_L'].notna()].copy()
            df_lidar = df.loc[valid_indices].copy()
            df_lidar.reset_index(drop=True, inplace=True)

            lidar_times = df_lidar['timestamp_sec'].values
            df_mouse['lidar_idx'] = np.digitize(df_mouse['timestamp_sec'], lidar_times)
            mouse_groups = df_mouse.groupby('lidar_idx')

            file_data = []

            for i in range(1, len(df_lidar)):
                if i not in mouse_groups.groups: continue

                prev_row = df_lidar.iloc[i-1]
                curr_row = df_lidar.iloc[i]

                g_dx = curr_row['lidar_x_mm'] - prev_row['lidar_x_mm']
                g_dy = curr_row['lidar_y_mm'] - prev_row['lidar_y_mm']
                dtheta_lidar = normalize_angle(curr_row['lidar_theta_rad'] - prev_row['lidar_theta_rad'])

                theta_prev = prev_row['lidar_theta_rad']
                dx_local = g_dx * np.cos(theta_prev) + g_dy * np.sin(theta_prev)
                dy_local = -g_dx * np.sin(theta_prev) + g_dy * np.cos(theta_prev)

                d_chord = np.sqrt(dx_local**2 + dy_local**2)

                if d_chord < MIN_STEP_DIST: continue

                cos_dth = np.cos(dtheta_lidar)
                sin_dth = np.sin(dtheta_lidar)

                GT_X_l = dx_local - N_OFFSET * (cos_dth - 1)
                GT_Y_l = dy_local - N_OFFSET * sin_dth

                GT_X_r = dx_local + N_OFFSET * (cos_dth - 1)
                GT_Y_r = dy_local + N_OFFSET * sin_dth

                m_interval = mouse_groups.get_group(i)
                nom_X_l_total, nom_Y_l_total = 0.0, 0.0
                nom_X_r_total, nom_Y_r_total = 0.0, 0.0
                D_total_l, D_total_r = 0.0, 0.0

                ticks = []

                for _, m in m_interval.iterrows():
                    nom_x_l = m['mouse_dx_L'] / CPI_LX
                    nom_y_l = -m['mouse_dy_L'] / CPI_LY
                    nom_x_r = m['mouse_dx_R'] / CPI_RX
                    nom_y_r = -m['mouse_dy_R'] / CPI_RY

                    d_l = np.sqrt(nom_x_l**2 + nom_y_l**2)
                    d_r = np.sqrt(nom_x_r**2 + nom_y_r**2)

                    ticks.append({
                        'nom_x_l': nom_x_l, 'nom_y_l': nom_y_l, 'd_l': d_l,
                        'nom_x_r': nom_x_r, 'nom_y_r': nom_y_r, 'd_r': d_r
                    })

                    nom_X_l_total += nom_x_l; nom_Y_l_total += nom_y_l
                    nom_X_r_total += nom_x_r; nom_Y_r_total += nom_y_r
                    D_total_l += d_l; D_total_r += d_r

                if D_total_l < 1e-6 or D_total_r < 1e-6: continue

                E_X_l = GT_X_l - nom_X_l_total
                E_Y_l = GT_Y_l - nom_Y_l_total
                E_X_r = GT_X_r - nom_X_r_total
                E_Y_r = GT_Y_r - nom_Y_r_total

                allocated_ticks_debug = []

                for idx, t in enumerate(ticks):
                    if t['d_l'] < 1e-6 and t['d_r'] < 1e-6:
                        continue

                    w_l = t['d_l'] / D_total_l if D_total_l > 0 else 0.0
                    w_r = t['d_r'] / D_total_r if D_total_r > 0 else 0.0

                    err_x_l = (E_X_l * w_l) if t['d_l'] > 1e-6 else 0.0
                    err_y_l = (E_Y_l * w_l) if t['d_l'] > 1e-6 else 0.0

                    err_x_r = (E_X_r * w_r) if t['d_r'] > 1e-6 else 0.0
                    err_y_r = (E_Y_r * w_r) if t['d_r'] > 1e-6 else 0.0

                    if not debug_printed:
                        allocated_ticks_debug.append({
                            'k': idx + 1,
                            'dl': t['d_l'], 'dr': t['d_r'],
                            'wl': w_l, 'wr': w_r,
                            'exl': err_x_l, 'eyl': err_y_l,
                            'exr': err_x_r, 'eyr': err_y_r
                        })

                    file_data.append({
                        'dist_x_mm_l': t['nom_x_l'] + 0.0,
                        'dist_y_mm_l': t['nom_y_l'] + 0.0,
                        'err_x_mm_l': err_x_l + 0.0,
                        'err_y_mm_l': err_y_l + 0.0,

                        'dist_x_mm_r': t['nom_x_r'] + 0.0,
                        'dist_y_mm_r': t['nom_y_r'] + 0.0,
                        'err_x_mm_r': err_x_r + 0.0,
                        'err_y_mm_r': err_y_r + 0.0
                    })

                if not debug_printed and allocated_ticks_debug:
                    print("\n" + "="*95)
                    print(f"KINEMATIC & ALLOCATION VERIFICATION (File: {file_name[:20]}...)")
                    print("="*95)
                    print("1. GLOBAL LiDAR DATA (100ms Step)")
                    print(f"   Start : X={prev_row['lidar_x_mm']:8.2f}, Y={prev_row['lidar_y_mm']:8.2f}, Th={prev_row['lidar_theta_rad']:6.3f} rad")
                    print(f"   End   : X={curr_row['lidar_x_mm']:8.2f}, Y={curr_row['lidar_y_mm']:8.2f}, Th={curr_row['lidar_theta_rad']:6.3f} rad")
                    print("-" * 95)
                    print("2. LOCAL CHASSIS FRAME (Rotated)")
                    print(f"   Local : dX={dx_local:8.2f}, dY={dy_local:8.2f}, dTh={dtheta_lidar:6.3f} rad")
                    print("-" * 95)
                    print("3. KINEMATIC GROUND TRUTH (Projected to Sensors)")
                    print(f"   Left Mouse GT  : dX={GT_X_l:8.2f}, dY={GT_Y_l:8.2f}")
                    print(f"   Right Mouse GT : dX={GT_X_r:8.2f}, dY={GT_Y_r:8.2f}")
                    print("-" * 95)
                    print(f"4. TOTAL MACROSCOPIC RESIDUAL ERROR (GT - Raw) [NO DEADBAND]")
                    print(f"   Left Error     : dX={E_X_l:8.2f}, dY={E_Y_l:8.2f}")
                    print(f"   Right Error    : dX={E_X_r:8.2f}, dY={E_Y_r:8.2f}")
                    print("-" * 95)
                    print("5. PHASE 3: PROPORTIONAL ALLOCATION (ALL 10ms TICKS IN BUCKET)")
                    print(f"{'Tick':<4} | {'d_L(mm)':<7} | {'w_L(%)':<7} | {'errX_L':<8} | {'errY_L':<8} || {'d_R(mm)':<7} | {'w_R(%)':<7} | {'errX_R':<8} | {'errY_R':<8}")
                    print("-" * 95)

                    sum_wl, sum_wr, sum_exl, sum_eyl, sum_exr, sum_eyr = 0, 0, 0, 0, 0, 0
                    for dt in allocated_ticks_debug:
                        print(f"{dt['k']:<4} | {dt['dl']:7.4f} | {dt['wl']*100:6.2f}% | {dt['exl']:8.4f} | {dt['eyl']:8.4f} || {dt['dr']:7.4f} | {dt['wr']*100:6.2f}% | {dt['exr']:8.4f} | {dt['eyr']:8.4f}")
                        sum_wl += dt['wl']
                        sum_wr += dt['wr']
                        sum_exl += dt['exl']
                        sum_eyl += dt['eyl']
                        sum_exr += dt['exr']
                        sum_eyr += dt['eyr']

                    print("-" * 95)
                    print(f"{'SUM':<4} | {'-':<7} | {sum_wl*100:6.2f}% | {sum_exl:8.4f} | {sum_eyl:8.4f} || {'-':<7} | {sum_wr*100:6.2f}% | {sum_exr:8.4f} | {sum_eyr:8.4f}")
                    print("="*95 + "\n")
                    debug_printed = True
                    print("Verification complete. Silently processing remaining files...")

            if file_data:
                out_path = os.path.join(OUTPUT_FOLDER, f"proc_{file_name}")
                pd.DataFrame(file_data).to_csv(out_path, index=False)

        except Exception as e:
            print(f"[ERROR] Failed to process {file_name}: {e}")

    print("\n" + "="*50)
    print("MASTER PROCESSING COMPLETE")
    print("="*50)
    print(f"Total Files Analyzed   : {len(csv_files)}")
    print(f"Valid Runs (Saved)  : {passed_count}")
    print(f"Skipped (<10cm)     : {skipped_count}")
    print(f"Processed files saved to ./{OUTPUT_FOLDER}/")
    print("="*50)

if __name__ == "__main__":
    process_master_dataset()