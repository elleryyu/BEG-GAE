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

def train_epoch(model, optimizer, device, graph, epoch):
    model.train()
    model = model.to(device)


    graph = graph.to(device)

    optimizer.zero_grad()

    reconstructed, embeddings = model(graph, graph.ndata['x'])

    loss = F.mse_loss(reconstructed, graph.ndata['x'])
    loss.backward()
    optimizer.step()
    epoch_loss = loss.detach().item()

    return epoch_loss, optimizer

if __name__ == '__main__':
    set_random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('saved_pkl/g_c3_site_16_1021_1.pkl', 'rb') as f:
        dataset = pickle.load(f)
        node_ids = pickle.load(f)

    print('node_ids:', node_ids)

    if isinstance(dataset, list):
        graph = dataset[0]
    else:
        graph = dataset

    in_feats = graph.ndata['x'].shape[1]
    hidden_feats = 64
    num_classes = in_feats

    model = GCN(in_feats=in_feats, hidden_feats=hidden_feats, num_classes=num_classes).to(device)
    model_path = 'saved_models_pth/gcn_model_single_graph_site_16_1028_0.pth'

    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded pre-trained model weights from {model_path}.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
    else:
        print(f"No pre-trained model found at {model_path}. Starting from scratch.")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 10000

    # Early stopping parameters
    patience = 3000  # Number of epochs with no improvement after which training stops
    min_delta = 0.0001  # Minimum change to qualify as an improvement
    best_loss = float('inf')
    counter = 0

    for epoch in range(1, num_epochs + 1):


        train_loss, optimizer = train_epoch(model, optimizer, device, graph, epoch)
        if epoch%100==0:
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"Training Loss: {train_loss:.4f}")

        # Early stopping logic
        if train_loss + min_delta < best_loss:
            best_loss = train_loss
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_path)
            # print(f"Training loss improved. Model saved to '{model_path}'.")
        else:
            counter += 1


        if counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    # After early stopping or full training, ensure the final model is saved
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to '{model_path}' after training.")
