import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"--- 2D Training Initiated on Device: {device} ---")

def kmeans_mps(data_tensor, k, iterations=20):
    n_samples, n_features = data_tensor.shape
    indices = torch.randperm(n_samples, device=device)[:k]
    centroids = data_tensor[indices].clone()

    for _ in range(iterations):
        distances = torch.cdist(data_tensor, centroids)
        cluster_idx = torch.argmin(distances, dim=1)

        new_centroids = torch.zeros((k, n_features), device=device)
        counts = torch.zeros((k, 1), device=device)

        cluster_idx_2d = cluster_idx.unsqueeze(1)
        new_centroids.scatter_add_(0, cluster_idx_2d.expand(-1, n_features), data_tensor)
        counts.scatter_add_(0, cluster_idx_2d, torch.ones((n_samples, 1), device=device))

        mask = (counts > 0).squeeze()
        new_centroids[mask] /= counts[mask]

        empty_mask = ~mask
        if empty_mask.any():
            new_centroids[empty_mask] = centroids[empty_mask]

        centroids = new_centroids
    return centroids

class RBFLayer(nn.Module):
    def __init__(self, centers, sigma):
        super(RBFLayer, self).__init__()
        self.centers = nn.Parameter(centers)
        self.sigma = nn.Parameter(sigma)

    def forward(self, x):
        distances = torch.cdist(x, self.centers)
        return torch.exp(-torch.pow(distances, 2) / (2 * torch.pow(self.sigma, 2)))

class RBFNN(nn.Module):
    def __init__(self, centers, sigma, out_features=2):
        super(RBFNN, self).__init__()
        self.rbf_layer = RBFLayer(centers, sigma)
        self.linear = nn.Linear(centers.shape[0], out_features)

    def forward(self, x):
        phi = self.rbf_layer(x)
        return self.linear(phi)

def train_mouse_model(X, y, mouse_side, k=30):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    print(f"\n[{mouse_side}] Initializing {k} centers via Native PyTorch KMeans in RAW mm space...")

    subset_idx = torch.randperm(X_train_t.size(0))[:min(200000, X_train_t.size(0))]
    X_subset = X_train_t[subset_idx]

    centers = kmeans_mps(X_subset, k)

    dists = torch.cdist(centers, centers)
    avg_dist = dists.sum() / (k**2 - k)
    sigma = torch.ones(k, dtype=torch.float32).to(device) * (avg_dist * 1.5)

    model = RBFNN(centers, sigma, out_features=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    criterion = nn.MSELoss()

    print(f"[{mouse_side}] Training on {len(X_train_t)} samples...")

    batch_size = 10000
    num_batches = max(len(X_train_t) // batch_size, 1)

    for epoch in range(1000):
        model.train()
        total_train_loss = torch.tensor(0.0, device=device)

        permutation = torch.randperm(X_train_t.size(0), device=device)

        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.detach()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t)

        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            avg_train = total_train_loss.item() / num_batches
            print(f"[{mouse_side}] Ep {epoch+1:3d} | Train MSE (mm^2): {avg_train:.6f} | Val MSE (mm^2): {val_loss.item():.6f} | LR: {optimizer.param_groups[0]['lr']:.5f}")
            if optimizer.param_groups[0]['lr'] < 1e-6:
                print(f"[{mouse_side}] Convergence reached.")
                break

    save_dir = f"model_weights_{mouse_side.lower()}"
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "centers.npy"), model.rbf_layer.centers.detach().cpu().numpy())
    np.save(os.path.join(save_dir, "sigma.npy"), model.rbf_layer.sigma.detach().cpu().numpy())
    np.save(os.path.join(save_dir, "weights.npy"), model.linear.weight.detach().cpu().numpy())
    np.save(os.path.join(save_dir, "bias.npy"), model.linear.bias.detach().cpu().numpy())
    print(f"Saved optimized, unscaled {k}-center model for {mouse_side} mouse.")

if __name__ == "__main__":
    folder = "processed_data_v2"
    files = glob.glob(os.path.join(folder, "proc_*.csv"))

    if not files:
        print(f"No CSV files found in '{folder}'.")
        exit()

    print(f"Loading {len(files)} files...")
    df_list = [pd.read_csv(f) for f in files]
    master_df = pd.concat(df_list, ignore_index=True)

    print("--- 2D Coupled Training with k=30 Initiated (UNFILTERED) ---")

    mask_L = (master_df['dist_x_mm_l'] != 0) | (master_df['dist_y_mm_l'] != 0)
    X_L = master_df.loc[mask_L, ['dist_x_mm_l', 'dist_y_mm_l']].values
    y_L = master_df.loc[mask_L, ['err_x_mm_l', 'err_y_mm_l']].values

    print(f"Filtered Left Mouse: {len(X_L)} moving samples.")
    train_mouse_model(X_L, y_L, "Left", k=30)

    mask_R = (master_df['dist_x_mm_r'] != 0) | (master_df['dist_y_mm_r'] != 0)
    X_R = master_df.loc[mask_R, ['dist_x_mm_r', 'dist_y_mm_r']].values
    y_R = master_df.loc[mask_R, ['err_x_mm_r', 'err_y_mm_r']].values

    print(f"Filtered Right Mouse: {len(X_R)} moving samples.")
    train_mouse_model(X_R, y_R, "Right", k=30)