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

# !Wrong
# def embedding_generation(model, device, data_loader):
#     model.eval()  # Set the model to evaluation mode
#     all_embeddings={}
#
#
#     with torch.no_grad():  # Disable gradient computation during evaluation
#             sample_i = next(iter(data_loader))
#             gs, subject_ids = sample_i
#             gs = gs.to(device)
#
#             # Forward pass: Get the reconstructed output and embeddings
#             _, embeddings = model(gs, gs.ndata['x'])
#             for idx, subject_id in enumerate(subject_ids):
#                 all_embeddings[subject_id] = embeddings[idx].cpu().numpy()
#
#     return all_embeddings

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

if __name__ == '__main__':
    # Set the device (GPU/CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load your Graph dataset
    with open('./c1_graph_site_16/all_graphs_1.pkl', 'rb') as f:

        dataset = pickle.load(f)

    # Define DataLoader (assuming batch_size is large enough to load the entire dataset at once)
    data_loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_func, shuffle=False)

    # Initialize your model (with the correct input/output dimensions)
    in_feats = 379  # Adjust to your dataset's input feature size
    hidden_feats = 128  # Define the hidden dimension size
    num_classes = in_feats  # For reconstruction, output the same number of features

    model = GCN(in_feats, hidden_feats, num_classes).to(device)

    # Load the pre-trained model weights (if needed)
    model.load_state_dict(torch.load('saved_models_pth/gcn_model_1021_1.pth'))

    # Generate embeddings for the entire dataset
    embeddings = embedding_generation(model, device, data_loader)

    # Save the embeddings to a pickle file
    with open('saved_pkl/graph_embeddings_site_16_1021.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

    print("Embeddings have been saved to 'graph_embeddings_site_16_1021.pkl'.")