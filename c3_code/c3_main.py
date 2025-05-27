import os
import torch

import pickle
from model import GCN
import torch.optim as optim
from torch.nn import functional as F
from utils import set_random_seed

"""

@Author: Weifeng Yu (Ellery)
@Created:  '2024-09-17'
@Last Modified:2025-03-23
@Email: dag9wj@virginia.edu

Description: This file is mainly for training population graph.

output: GAE_pop.model.pth
"""
# ==== Configuration ====
GRAPH_PATH = 'saved_pkl/g_c3_site_16_1021_1.pkl'
MODEL_PATH = 'saved_models_pth/gcn_model_single_graph_site_16_1028_0.pth'
IN_FEATS = None  # Will be inferred from graph
HIDDEN_FEATS = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10000
PATIENCE = 3000
MIN_DELTA = 1e-4
SEED = 42


# ==== Training Function ====
def train_epoch(model, optimizer, device, graph):
    model.train()
    model.to(device)
    graph = graph.to(device)

    optimizer.zero_grad()
    reconstructed, _ = model(graph, graph.ndata['x'])
    loss = F.mse_loss(reconstructed, graph.ndata['x'])
    loss.backward()
    optimizer.step()

    return loss.item()


# ==== Main Execution ====
def main():
    # Set random seed and device
    set_random_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graph and node IDs
    with open(GRAPH_PATH, 'rb') as f:
        dataset = pickle.load(f)
        node_ids = pickle.load(f)

    print(f"Loaded node IDs: {node_ids[:10]}...")

    graph = dataset[0] if isinstance(dataset, list) else dataset

    # Define model dimensions
    in_feats = graph.ndata['x'].shape[1]
    model = GCN(in_feats=in_feats, hidden_feats=HIDDEN_FEATS, num_classes=in_feats).to(device)

    # Load pre-trained weights if available
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            print(f"Loaded pre-trained model from: {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print(f"No pre-trained model found at {MODEL_PATH}. Starting from scratch.")

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Early stopping setup
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_epoch(model, optimizer, device, graph)

        # Log every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{NUM_EPOCHS} - Training Loss: {train_loss:.6f}")

        # Check for improvement
        if train_loss + MIN_DELTA < best_loss:
            best_loss = train_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            epochs_without_improvement += 1

        # Trigger early stopping
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered after {PATIENCE} epochs without improvement.")
            break

    # Final model save
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Final model saved to: {MODEL_PATH}")


if __name__ == '__main__':
    main()