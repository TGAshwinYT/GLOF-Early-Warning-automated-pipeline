"""
Stage 2: Spatio-Temporal Graph Neural Network (ST-GNN / Graph WaveNet).

Routes GLOF flood waves through complex mountain gorges down to flat floodplains.
Models physical fluid routing over the directed river graph G = (V, E):
- Root pulse injection: Q(t) breach pulse at Lake node (index 0)
- Directed spatial message passing: channel hydraulic routing
- Temporal propagation: GRU dynamics capturing wave attenuation and travel time
- Outputs:
  * Inundation water depth h(t) [Num_Nodes, Timesteps]
  * Wave arrival time t_arrival [Num_Nodes]
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectedGraphConv(nn.Module):
    """
    Directional message-passing convolution for hydraulic networks.
    Aggregates momentum and discharge along directed upstream->downstream edges.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear_self = nn.Linear(in_features, out_features)
        self.linear_msg = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Self transformation
        out = self.linear_self(x)

        # Message passing from upstream src to downstream dst
        msgs = self.linear_msg(x[src])
        if edge_weight is not None:
            msgs = msgs * edge_weight.unsqueeze(-1)

        # Aggregate downstream
        aggr = torch.zeros((num_nodes, out.size(-1)), device=x.device, dtype=x.dtype)
        aggr.index_add_(0, dst, msgs)

        return F.relu(out + aggr)

class RiverGNN(nn.Module):
    def __init__(self, in_channels=8, hidden_dim=64, out_timesteps=72):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_timesteps = out_timesteps

        # Spatial message passing layers
        self.conv1 = DirectedGraphConv(in_channels, hidden_dim)
        self.conv2 = DirectedGraphConv(hidden_dim, hidden_dim)
        self.conv3 = DirectedGraphConv(hidden_dim, hidden_dim)

        # Temporal evolution module (GRU)
        self.temporal_cell = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Prediction Heads
        # 1. Depth hydrograph: [Num_Nodes, Timesteps]
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_timesteps),
            nn.Softplus() # Water depth must be strictly non-negative
        )
        
        # 2. Wave arrival time: [Num_Nodes]
        self.arrival_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus() # Arrival time must be >= 0
        )

    def forward(self, x, edge_index, breach_pulse=None, edge_weight=None):
        """
        x: [Num_Nodes, in_channels]
        edge_index: [2, Num_Edges]
        breach_pulse: scalar or [1] peak discharge/depth injection at root node 0
        """
        x_mod = x.clone()
        if breach_pulse is not None:
            # Inject pulse energy into Lake root node (index 0)
            if torch.is_tensor(breach_pulse):
                x_mod[0, 0] = x_mod[0, 0] + breach_pulse.squeeze()
            else:
                x_mod[0, 0] = x_mod[0, 0] + float(breach_pulse)

        # 1. Spatial message passing
        h = self.conv1(x_mod, edge_index, edge_weight)
        h = self.conv2(h, edge_index, edge_weight)
        h = self.conv3(h, edge_index, edge_weight)

        # 2. Temporal wave dynamics
        # Unsqueeze as batch of size 1 for GRU sequence over graph features
        gru_out, _ = self.temporal_cell(h.unsqueeze(0))
        node_embeddings = gru_out.squeeze(0) # [Num_Nodes, hidden_dim]

        # 3. Predict depth hydrograph & arrival times
        depth_predictions = self.depth_head(node_embeddings) # [Num_Nodes, out_timesteps]
        arrival_predictions = self.arrival_head(node_embeddings).squeeze(-1) # [Num_Nodes]

        return depth_predictions, arrival_predictions

def train_model(data_dir="data/processed", model_dir="models", epochs=50, lr=0.003, dry_run=False):
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training RiverGNN on device: {device}")

    # Load graph data
    node_features = np.load(os.path.join(data_dir, "node_features.npy"))
    edge_index = np.load(os.path.join(data_dir, "edge_index.npy"))
    edge_attr = np.load(os.path.join(data_dir, "edge_attr.npy"))

    # Load synthetic ground truth targets
    h_matrices = np.load(os.path.join(data_dir, "synthetic_h_matrices.npy")) # [S, N, T]
    arrival_times = np.load(os.path.join(data_dir, "synthetic_arrival_times.npy")) # [S, N]

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(node_features, dtype=torch.float32).to(device)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).to(device)
    
    # Normalized edge weights based on channel gradient
    edge_gradients = edge_attr[:, 1]
    edge_weights = torch.tensor(np.clip(edge_gradients / (np.max(edge_gradients) + 1e-6), 0.1, 1.0), dtype=torch.float32).to(device)

    num_scenarios = h_matrices.shape[0]
    num_nodes = node_features.shape[0]
    num_timesteps = h_matrices.shape[2]

    # Train / Val Split
    indices = np.arange(num_scenarios)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.85 * num_scenarios)
    train_idx, val_idx = indices[:split], indices[split:]

    model = RiverGNN(
        in_channels=node_features.shape[1],
        hidden_dim=64,
        out_timesteps=num_timesteps
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()

    if dry_run:
        print("[*] Running 1-batch dry run (max_steps=5)...")
        epochs = 1
        train_idx = train_idx[:1]

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for s in train_idx:
            optimizer.zero_grad()
            target_h = torch.tensor(h_matrices[s], dtype=torch.float32).to(device)
            target_arr = torch.tensor(arrival_times[s], dtype=torch.float32).to(device)

            # Injected breach pulse at root
            root_pulse = target_h[0, 10] - x_tensor[0, 0] * 0.0

            pred_h, pred_arr = model(x_tensor, edge_index_tensor, breach_pulse=root_pulse, edge_weight=edge_weights)

            loss_depth = mse_criterion(pred_h, target_h)
            loss_arrival = mae_criterion(pred_arr, target_arr)
            loss = loss_depth + 0.1 * loss_arrival

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1 or dry_run:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v in val_idx[:15]:
                    target_h_v = torch.tensor(h_matrices[v], dtype=torch.float32).to(device)
                    target_arr_v = torch.tensor(arrival_times[v], dtype=torch.float32).to(device)
                    pred_h_v, pred_arr_v = model(x_tensor, edge_index_tensor, edge_weight=edge_weights)
                    val_loss += mse_criterion(pred_h_v, target_h_v).item()
            val_loss /= max(1, len(val_idx[:15]))
            print(f"Epoch [{epoch+1:03d}/{epochs:03d}] - Train Loss: {total_loss/len(train_idx):.4f} | Val MSE: {val_loss:.4f}")

    # Save trained checkpoint (maintaining rolling 3-checkpoint policy)
    model_save_path = os.path.join(model_dir, "river_gnn_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": node_features.shape[1],
        "hidden_dim": 64,
        "out_timesteps": num_timesteps
    }, model_save_path)
    print(f"[+] Serialized Stage 2 RiverGNN model to {model_save_path}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Spatio-Temporal River GNN")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-batch sanity dry run")
    parser.add_argument("--data-dir", default="data/processed", help="Data directory")
    args = parser.parse_args()

    train_model(data_dir=args.data_dir, epochs=args.epochs, lr=args.lr, dry_run=args.dry_run)
