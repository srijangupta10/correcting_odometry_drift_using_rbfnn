import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans

DATA_DIR = "processed_data_v2"
SAMPLE_SIZE = 200000
K_RANGE = range(2, 42, 2)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def load_and_sample_data():
    all_files = glob.glob(os.path.join(DATA_DIR, "proc_*.csv"))
    if not all_files:
        print(f"Error: No processed files found in {DATA_DIR}")
        return None, None
    
    print(f"Loading data from {len(all_files)} files...")
    df_list = [pd.read_csv(f) for f in all_files]
    full_df = pd.concat(df_list, ignore_index=True)
    
    print(f"Total rows loaded: {len(full_df)}")
    
    if len(full_df) > SAMPLE_SIZE:
        print(f"Subsampling {SAMPLE_SIZE} random rows for Elbow Analysis...")
        df_sampled = full_df.sample(n=SAMPLE_SIZE, random_state=42)
    else:
        df_sampled = full_df
        
    X_left = df_sampled[['dist_x_mm_l', 'dist_y_mm_l']].values
    X_right = df_sampled[['dist_x_mm_r', 'dist_y_mm_r']].values
    
    return X_left, X_right

def run_elbow(X_left, X_right):
    inertia_l = []
    inertia_r = []
    
    print("\nRunning MiniBatch K-Means for LEFT Mouse...")
    for k in K_RANGE:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10000, n_init=1)
        kmeans.fit(X_left)
        inertia_l.append(kmeans.inertia_)
        print(f"   K={k:2d} | Inertia={kmeans.inertia_:.2f}")
        
    print("\nRunning MiniBatch K-Means for RIGHT Mouse...")
    for k in K_RANGE:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=10000, n_init=1)
        kmeans.fit(X_right)
        inertia_r.append(kmeans.inertia_)
        print(f"   K={k:2d} | Inertia={kmeans.inertia_:.2f}")

    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(K_RANGE, inertia_l, marker='o', linestyle='-', color='blue', linewidth=2)
    plt.title('Left Mouse - Kinematic States (Elbow Curve)', fontsize=14)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia (Sum of Squared Distances in mm²)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(K_RANGE)
    
    plt.subplot(1, 2, 2)
    plt.plot(K_RANGE, inertia_r, marker='o', linestyle='-', color='red', linewidth=2)
    plt.title('Right Mouse - Kinematic States (Elbow Curve)', fontsize=14)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Inertia (Sum of Squared Distances in mm²)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(K_RANGE)
    
    plt.tight_layout()
    plot_path = 'elbow_curve_mm.png'
    plt.savefig(plot_path, dpi=300)
    print(f"\nAnalysis complete! Saved plot as '{plot_path}'")
    plt.show()

if __name__ == "__main__":
    X_l, X_r = load_and_sample_data()
    if X_l is not None:
        run_elbow(X_l, X_r)