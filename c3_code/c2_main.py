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


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    set_seed(42)

    X = pd.read_csv('site_16/ABCD_site_16_additional_scores.csv', header=0, index_col=0)

    y = pd.read_csv('site_16/site_16_labels.csv', index_col=0)


    ##################################################
    # X = X.dropna(axis=1)
    # # X=X.loc[selected_subject_ids]
    # original_indices = X.index.tolist()
    # X = torch.tensor(X.values)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # X = torch.tensor(X, dtype=torch.float32)
    # print(X)
    # num_nodes = X.shape[0]
    # feature_dim = X.shape[1]
    # g = dgl.graph(([], []), num_nodes=num_nodes)
    # g.ndata['x'] = X.float()
    #
    # print(g)
    # feature_dim = X.shape[1]
    #
    # model = WeightModel(feature_dim=feature_dim)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #
    # num_epochs = 1
    #
    # patience = 50
    # best_loss = np.inf
    # trigger_times = 0
    # best_model = None
    #
    # for epoch in range(num_epochs):
    #     model.train()
    #     optimizer.zero_grad()
    #
    #     E = model(X)
    #
    #     loss = custom_loss(E, X, alpha=0.1)
    #
    #     loss.backward()
    #     optimizer.step()
    #
    #     # Early stopping
    #     current_loss = loss.item()
    #     if current_loss < best_loss:
    #         best_loss = current_loss
    #         trigger_times = 0
    #         best_model = model.state_dict()
    #     else:
    #         trigger_times += 1
    #         if trigger_times >= patience:
    #             print(f"Early stopping triggered at epoch {epoch + 1}")
    #             model.load_state_dict(best_model)
    #             break
    #
    #     if (epoch + 1) % 100 == 0:
    #         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {current_loss:.4f}')
    #
    # E_np = E.detach().numpy()
    # E_np_without_binary = E_np.copy()
    # threshold = 6
    # E_np = np.where(E_np > threshold, 0, 1)
    #
    # plt.imshow(1 / E_np_without_binary, cmap='hot', interpolation='nearest')
    # plt.title('Learned Adjacency Matrix')
    # plt.colorbar()
    # plt.show()
    # degrees = np.sum(E_np, axis=1)
    # outliers = np.where(degrees < 1e-3)[0]
    # print(f'Outliers (nodes with near-zero connections): {outliers}')
    # print(f'Number of outliers: {len(outliers)}')
    #
    # plt.hist(degrees, bins=30, color='blue', alpha=0.7)
    # plt.title('Node Degree Distribution')
    # plt.xlabel('Node Degree')
    # plt.ylabel('Frequency')
    # plt.show()
    # adjacency_matrix_with_indices = pd.DataFrame(E_np, index=original_indices, columns=original_indices)
    # # -----------------------------------------
    # total_elements = E_np.size
    # ones_count = np.sum(E_np == 1)
    # sparsity = ones_count / total_elements
    #
    # print(f'Sparsity of the adjacency matrix (proportion of 1s): {sparsity: .4f}')
    # print(f'Number of 1 elements: {ones_count}, Total elements: {total_elements}')
    # print(adjacency_matrix_with_indices)
    #
    # c3_graph = Graph_with_adjacency_matrix(adjacency_matrix=adjacency_matrix_with_indices,
    #                                        embedding_path='saved_pkl/graph_embeddings.pkl')
    # g_c3, node_ids = c3_graph[0]
    # print(g_c3.ndata['y'])
    # print(g_c3)
    #
    # with open('saved_pkl/g_c3.pkl', 'wb') as f:
    #     pickle.dump(g_c3, f)
    #     pickle.dump(node_ids,f)
    #
    # print("g_c3 has been saved to 'g_c3.pkl'.")
    ###########################################
    X = X.dropna(axis=1)
    print(f"Shape of X after dropping NaNs: {X.shape}")

    # Get the indices (subject IDs)
    original_indices = X.index.tolist()
    print(f"Indices of X:\n{X.index}")

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to PyTorch Tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    print(f"Shape of X_tensor: {X_tensor.shape}")

    # Compute cosine similarity
    cosine_sim = cosine_similarity(X_scaled)

    # Create adjacency matrix based on a threshold
    threshold = 0.55  # Adjust threshold as needed
    adj_matrix = np.where(cosine_sim > threshold, 1, 0)
    np.fill_diagonal(adj_matrix, 0)

    # Visualize the adjacency matrix
    plt.imshow(adj_matrix, cmap='hot', interpolation='nearest')
    plt.title('Cosine Similarity Adjacency Matrix')
    plt.colorbar()
    plt.show()

    # Compute node degrees
    degrees = np.sum(adj_matrix, axis=1)
    outliers = np.where(degrees < 1e-3)[0]
    print(f'Outliers (nodes with near-zero connections): {outliers}')
    print(f'Number of outliers: {len(outliers)}')

    # Plot node degree distribution
    plt.hist(degrees, bins=30, color='blue', alpha=0.7)
    plt.title('Node Degree Distribution')
    plt.xlabel('Node Degree')
    plt.ylabel('Frequency')
    plt.show()

    # Create a DataFrame for the adjacency matrix with indices
    adjacency_matrix_with_indices = pd.DataFrame(adj_matrix, index=original_indices, columns=original_indices)

    # Calculate sparsity
    total_elements = adj_matrix.size
    ones_count = np.sum(adj_matrix == 1)
    sparsity = ones_count / total_elements

    print(f'Sparsity of the adjacency matrix (proportion of 1s): {sparsity: .4f}')
    print(f'Number of 1 elements: {ones_count}, Total elements: {total_elements}')
    print(f'Adjacency matrix with indices:\n{adjacency_matrix_with_indices}')

    # Create the graph using your custom Graph_with_adjacency_matrix class
    c3_graph = Graph_with_adjacency_matrix(
        adjacency_matrix=adjacency_matrix_with_indices,
        embedding_path='saved_pkl/graph_embeddings_site_16_1028_max.pkl',
        label_path='site_16/site_16_2yrFL_ocd_anx_labels_with_specific_phobia.csv'
    )
    g_c3, node_ids = c3_graph[0]
    print(f"Node labels:\n{g_c3.ndata['y']}")
    print(f"Graph details:\n{g_c3}")

    # Save the graph and node IDs
    with open('saved_pkl/g_c3_site_16_1028_max_2yrFL_with_sph.pkl', 'wb') as f:
        pickle.dump(g_c3, f)
        pickle.dump(node_ids, f)

    print("Graph 'g_c3' has been saved to 'g_c3_site_16_1028_max_2yrFL_with_sph.pkl'.")