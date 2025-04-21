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


if __name__ == '__main__':
    set_random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ratio=0.7
    val_ratio=0.2

    with open('./c1_graph_site_16/all_graphs_1.pkl', 'rb') as f:
        dataset = pickle.load(f)


    train_size=int(train_ratio*len(dataset))
    val_size=int(val_ratio*len(dataset))
    test_size= len(dataset)-train_size-val_size

    train_set, val_set, test_set=torch.utils.data.random_split(dataset,[train_size,val_size,test_size])

    train_loader= DataLoader(train_set,shuffle=True,batch_size=16,collate_fn=collate_func)
    val_loader= DataLoader(val_set,batch_size=4,collate_fn=collate_func)
    test_loader=DataLoader(test_set,batch_size=4,collate_fn=collate_func)

    in_feats = 379  # Input feature size (based on your graph size)
    hidden_feats = 128  # Hidden layer size (can be tuned)
    num_classes = in_feats  # For reconstruction, output the same number of features

    model = GCN(in_feats, hidden_feats, num_classes).to(device)

    model_path='saved_models_pth/gcn_model_1021_1.pth'

    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded pre-trained model weights from {model_path}.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print(f"No pre-trained model found at {model_path}. Starting from scratch.")


    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Set number of epochs
    num_epochs = 80

    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        # Train for one epoch
        train_loss, optimizer = train_epoch(model, optimizer, device, train_loader, epoch)

        # Evaluate on the validation set
        val_loss, val_embeddings = evaluate_network(model, device, val_loader, epoch)
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    test_loss, test_embeddings = evaluate_network(model, device, test_loader, epoch=1)
    print(f"Testing Loss: {test_loss:.4f}")
    # Save the final trained model
    torch.save(model.state_dict(), 'saved_models_pth/gcn_model_1021_1.pth')
    print("Model saved to './gcn_model_1021_1.pth'.")
