# ML-Based Guillotine Bin Packing with Coordinate Prediction, Rotation, and Visualization
# Author: Abuzar B. M. Adam
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Configuration
BIN_WIDTH = 100
BIN_HEIGHT = 100
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.001

# Dataset
class BinPackingDataset(Dataset):
    def __init__(self, df, scaler=None):
        self.features = df[["item_width", "item_height", "aspect_ratio", "area", "fill_ratio"]].values
        self.targets_xy = df[["x", "y"]].values / np.array([BIN_WIDTH, BIN_HEIGHT])  # normalize coords
        self.targets_rot = df["rotation"].values
        self.feasible = df["feasible"].values
        if scaler:
            self.features = scaler.transform(self.features)
        self.X = torch.tensor(self.features, dtype=torch.float32)
        self.y_xy = torch.tensor(self.targets_xy, dtype=torch.float32)
        self.y_rot = torch.tensor(self.targets_rot, dtype=torch.long)
        self.feasible = torch.tensor(self.feasible, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_xy[idx], self.y_rot[idx], self.feasible[idx]

# Model
class PackingNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.coord_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        self.rot_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        h = self.shared(x)
        coords = self.coord_head(h)
        rotation_logits = self.rot_head(h)
        return coords, rotation_logits

# Load and preprocess dataset
df = pd.read_csv(".../data/guillotine_binpacking_dataset.csv")
features = df[["item_width", "item_height", "aspect_ratio", "area", "fill_ratio"]].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train/test split
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)
train_set = BinPackingDataset(df_train, scaler)
val_set = BinPackingDataset(df_val, scaler)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PackingNet(input_dim=5).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion_coord = nn.MSELoss()
criterion_rot = nn.CrossEntropyLoss()

train_loss_log = []
val_loss_log = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X, y_xy, y_rot, feasible in train_loader:
        X, y_xy, y_rot = X.to(device), y_xy.to(device), y_rot.to(device)
        pred_xy, pred_rot = model(X)
        loss_coord = criterion_coord(pred_xy, y_xy)
        loss_rot = criterion_rot(pred_rot, y_rot)
        loss = loss_coord + loss_rot
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_loss_log.append(total_loss / len(train_loader))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y_xy, y_rot, feasible in val_loader:
            X, y_xy, y_rot = X.to(device), y_xy.to(device), y_rot.to(device)
            pred_xy, pred_rot = model(X)
            loss = criterion_coord(pred_xy, y_xy) + criterion_rot(pred_rot, y_rot)
            val_loss += loss.item()
    val_loss_log.append(val_loss / len(val_loader))
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss_log[-1]:.4f}, Val Loss: {val_loss_log[-1]:.4f}")

# Save
torch.save(model.state_dict(), ".../model/binpacking_model.pt")
joblib.dump(scaler, ".../model/binpacking_scaler.pkl")

# Inference & Simulation
def simulate_bin_packing(model, scaler, n_items=10):
    model.eval()
    bin_rect = [0, 0, BIN_WIDTH, BIN_HEIGHT]
    free_rects = [bin_rect]  # list of [x, y, w, h]
    placed = []
    base_cmap = plt.get_cmap('tab10')
    colors = [base_cmap(i % base_cmap.N) for i in range(n_items)]

    for i in range(n_items):
        item_w = np.random.uniform(10, 40)
        item_h = np.random.uniform(10, 40)
        ar = item_w / item_h
        area = item_w * item_h
        fill_ratio = area / (BIN_WIDTH * BIN_HEIGHT)
        feature = np.array([[item_w, item_h, ar, area, fill_ratio]])
        x_in = torch.tensor(scaler.transform(feature), dtype=torch.float32).to(device)
        with torch.no_grad():
            coords, rot_logits = model(x_in)
            rot = torch.argmax(rot_logits, dim=1).item()

        iw, ih = (item_h, item_w) if rot == 1 else (item_w, item_h)

        # Try placing item in all free rects (greedy + overlap check)
        fits = False
        for fr in free_rects:
            x0, y0, fw, fh = fr
            if iw <= fw and ih <= fh:
                px, py = x0, y0

                # Check overlap with already placed items
                overlap = False
                for (ox, oy, ow, oh, _) in placed:
                    if not (px + iw <= ox or px >= ox + ow or py + ih <= oy or py >= oy + oh):
                        overlap = True
                        break

                if not overlap:
                    fits = True
                    free_rects.remove(fr)
                    free_rects.append([x0, y0, fw, py - y0])  # top
                    free_rects.append([x0, py + ih, fw, y0 + fh - (py + ih)])  # bottom
                    free_rects.append([x0, y0, px - x0, fh])  # left
                    free_rects.append([px + iw, y0, x0 + fw - (px + iw), fh])  # right
                    break

        if fits:
            placed.append((px, py, iw, ih, rot))

    # Visualization
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, BIN_WIDTH)
    ax.set_ylim(0, BIN_HEIGHT)
    for i, (x, y, w, h, r) in enumerate(placed):
        rect = plt.Rectangle((x, y), w, h, edgecolor='black', facecolor=colors[i], alpha=0.7)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, f"{i}", ha='center', va='center')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(".../model/bin_packing_result.png")
    plt.close()
    print(f"Placed {len(placed)} out of {n_items} items.")

# Run simulation
if __name__ == "__main__":
    model.load_state_dict(torch.load(".../model/binpacking_model.pt", map_location=device))
    simulate_bin_packing(model, scaler)
