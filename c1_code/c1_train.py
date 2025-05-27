"""
Author: Weifeng Yu (Ellery)
Created:  '2024-09-01'
Last Modified:2025-04-26
Email: dag9wj@virginia.edu
"""
import os
import torch
import torch.nn as nn
import pickle
from model import GCN  # Assuming you have defined GCN model in model.py
import dgl
from tqdm import tqdm  # Import tqdm for progress bar
import random
import  numpy as np
import torch.optim as optim
from utils import collate_func,set_random_seed
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import FC_Graph

# ==== Configuration ====
GRAPH_PATH = './c1_graph_site_16/all_graphs_1.pkl'
MODEL_PATH = 'saved_models_pth/gcn_model_1021_1.pth'
BATCH_SIZE = 16
IN_FEATURES = 379
HIDDEN_FEATURES = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 80
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
SEED = 42



def train_epoch(model, optimizer, device, data_loader, epoch):
    model.train()
    model = model.to(device)
    epoch_loss = 0

    for iter, sample_i in enumerate(tqdm(data_loader, desc="Training iterations!")):
        optimizer.zero_grad()
        gs,subject_ids = sample_i
        gs = gs.to(device)

        # Forward pass: Get the reconstructed output and embeddings
        reconstructed, embeddings = model(gs, gs.ndata['x'])

        # Reconstruction loss (MSE)
        loss = F.mse_loss(reconstructed, gs.ndata['x'])

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        epoch_loss += loss.detach().item()

    epoch_loss /= (iter + 1)
    return epoch_loss, optimizer

def evaluate_network(model, device, data_loader, epoch):
    model.eval()  # Set the model to evaluation mode
    epoch_loss = 0

    with torch.no_grad():  # Disable gradient computation during evaluation
        for iter, sample_i in enumerate(tqdm(data_loader, desc="Evaluation iterations!")):
            gs, subject_ids = sample_i
            gs = gs.to(device)

            # Forward pass: Get the reconstructed output and embeddings
            reconstructed, embeddings = model(gs, gs.ndata['x'])

            # Compute reconstruction loss (MSE)
            loss = F.mse_loss(reconstructed, gs.ndata['x'])

            # Accumulate the loss
            epoch_loss += loss.detach().item()

    epoch_loss /= (iter + 1)
    return epoch_loss, embeddings


# ==== Main Script ====
def main():
    set_random_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    with open(GRAPH_PATH, 'rb') as f:
        dataset = pickle.load(f)

    # Split dataset
    total_size = len(dataset)
    train_size = int(TRAIN_RATIO * total_size)
    val_size = int(VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

    # Define DataLoaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_func)
    test_loader = DataLoader(test_set, batch_size=test_size, shuffle=False, collate_fn=collate_func)

    # Initialize model
    model = GCN(IN_FEATURES, HIDDEN_FEATURES, IN_FEATURES).to(device)

    # Load pre-trained weights if available
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            print(f"Loaded pre-trained model from: {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load pre-trained model. Error: {e}")
    else:
        print(f"No pre-trained model found at {MODEL_PATH}. Training from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        train_loss, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)
        val_loss, _ = evaluate_network(model, device, val_loader, epoch)
        print(f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}")

    # Final evaluation on test set
    test_loss, test_embeddings = evaluate_network(model, device, test_loader, epoch=NUM_EPOCHS + 1)
    print(f"\nFinal Test Loss: {test_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Trained model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()