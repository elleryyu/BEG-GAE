import torch
import pandas as pd
import  numpy as np
from model import WeightModel
from sklearn.preprocessing import StandardScaler
from utils import custom_loss
import dgl
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from data import Graph_with_adjacency_matrix

"""

@Author: Weifeng Yu (Ellery)
@Created:  '2024-09-17'
@Last Modified:2025-03-23
@Email: dag9wj@virginia.edu

Description: This file contains 2 module: Generate the X: features, generate W using EdgeMode from model.py
if Matrix information exists, this files main will update model instead of training from the beginning.
The subject we chose are site_11 from ABCD and contains functinoal connectivity information


output: graph that contains behavioral scored adj matrix and c1 embeddings
"""

# ==== Configuration ====
SEED = 42
SCORE_PATH = 'site_16/ABCD_site_16_additional_scores.csv'
LABEL_PATH = 'site_16/site_16_labels.csv'
EMBEDDING_PATH = 'saved_pkl/graph_embeddings_site_16_1028_max.pkl'
SPECIFIC_LABEL_PATH = 'site_16/site_16_2yrFL_ocd_anx_labels_with_specific_phobia.csv'
SAVE_PATH = 'saved_pkl/g_c3_site_16_1028_max_2yrFL_with_sph.pkl'
SIMILARITY_THRESHOLD = 0.55


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Set random seed
    set_seed(SEED)

    # Load data
    X = pd.read_csv(SCORE_PATH, header=0, index_col=0)
    y = pd.read_csv(LABEL_PATH, index_col=0)

    # Drop NaN columns
    X = X.dropna(axis=1)
    print(f"Shape of X after dropping NaNs: {X.shape}")

    # Preserve subject indices
    original_indices = X.index.tolist()
    print(f"Subject indices (X):\n{X.index}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    print(f"Shape of X_tensor: {X_tensor.shape}")

    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(X_scaled)

    # Thresholding to build adjacency matrix
    adj_matrix = np.where(cosine_sim > SIMILARITY_THRESHOLD, 1, 0)
    np.fill_diagonal(adj_matrix, 0)

    # Visualize adjacency matrix
    plt.imshow(adj_matrix, cmap='hot', interpolation='nearest')
    plt.title('Cosine Similarity Adjacency Matrix')
    plt.colorbar()
    plt.show()

    # Compute node degree and outliers
    degrees = np.sum(adj_matrix, axis=1)
    outliers = np.where(degrees < 1e-3)[0]
    print(f'Outliers (nodes with near-zero connections): {outliers.tolist()}')
    print(f'Number of outliers: {len(outliers)}')

    # Plot node degree histogram
    plt.hist(degrees, bins=30, color='blue', alpha=0.7)
    plt.title('Node Degree Distribution')
    plt.xlabel('Node Degree')
    plt.ylabel('Frequency')
    plt.show()

    # Wrap adjacency matrix with indices
    adjacency_df = pd.DataFrame(adj_matrix, index=original_indices, columns=original_indices)

    # Compute and report sparsity
    sparsity = np.sum(adj_matrix == 1) / adj_matrix.size
    print(f'Sparsity of adjacency matrix (proportion of 1s): {sparsity:.4f}')
    print(f'Number of 1s: {np.sum(adj_matrix == 1)}, Total elements: {adj_matrix.size}')
    print("Sample of adjacency matrix with indices:")
    print(adjacency_df.iloc[:5, :5])

    # Construct graph object
    graph_data = Graph_with_adjacency_matrix(
        adjacency_matrix=adjacency_df,
        embedding_path=EMBEDDING_PATH,
        label_path=SPECIFIC_LABEL_PATH
    )
    g_c3, node_ids = graph_data[0]

    print("Graph construction complete.")
    print(f"Node labels sample:\n{g_c3.ndata['y'][:10]}")
    print(f"Graph summary:\n{g_c3}")

    # Save graph and node IDs
    with open(SAVE_PATH, 'wb') as f:
        pickle.dump(g_c3, f)
        pickle.dump(node_ids, f)

    print(f"Graph has been saved to '{SAVE_PATH}'.")

if __name__ == '__main__':
    main()