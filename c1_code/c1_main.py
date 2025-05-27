"""
Author: Weifeng Yu (Ellery)
Created:  '2024-09-01'
Last Modified:2025-04-17
Email: dag9wj@virginia.edu
"""

import torch
import pickle
from model import GCN  # Assuming you have defined GCN model in model.py
from utils import collate_func
from torch.nn import functional as F
from torch.utils.data import DataLoader
from data import FC_Graph

# ==== Configuration ====
GRAPH_PATH = './c1_graph_site_16/all_graphs_1.pkl'
MODEL_PATH = 'saved_models_pth/gcn_model_1021_1.pth'
OUTPUT_PATH = 'saved_pkl/graph_embeddings_site_16_1021.pkl'

IN_FEATURES = 379
HIDDEN_FEATURES = 128
BATCH_SIZE = None  # Set to None to use len(dataset) dynamically


def embedding_generation(model, device, data_loader):
    model.eval()  # Set the model to evaluation mode
    all_embeddings = {}
    all_graph_embeddings = {}  

    with torch.no_grad():  # Disable gradient computation during evaluation
        sample_i = next(iter(data_loader))
        gs, subject_ids = sample_i
        gs = gs.to(device)

        # Forward pass: Get the reconstructed output and embeddings
        _, embeddings = model(gs, gs.ndata['x'])

        # Reshape embeddings to [440, 379, 128]
        num_graphs = 440
        num_nodes_per_graph = 379
        embeddings = embeddings.view(num_graphs, num_nodes_per_graph, -1)

        # Store embeddings
        for graph_idx, subject_id in enumerate(subject_ids):
            # Store all node embeddings for this graph
            all_embeddings[subject_id] = embeddings[graph_idx].cpu().numpy()

            # Compute the graph-level embedding using max pooling across the node dimension
            graph_embedding = embeddings[graph_idx].max(dim=0).values  # Shape: [128]
            all_graph_embeddings[subject_id] = graph_embedding.cpu().numpy()

        return all_graph_embeddings

def main():
    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    with open(GRAPH_PATH, 'rb') as f:
        dataset = pickle.load(f)

    # Define DataLoader
    loader = DataLoader(
        dataset,
        batch_size=len(dataset) if BATCH_SIZE is None else BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_func
    )

    # Initialize model
    model = GCN(IN_FEATURES, HIDDEN_FEATURES, IN_FEATURES).to(device)

    # Load pre-trained weights
    model.load_state_dict(torch.load(MODEL_PATH))

    # Generate embeddings
    embeddings = embedding_generation(model, device, loader)

    # Save embeddings
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(embeddings, f)

    print(f"Embeddings saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()