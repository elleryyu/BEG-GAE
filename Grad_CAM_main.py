"""
Author: Gang Qu
Created:  '2024-09-01'
Last Modified:2025-04-26
Email:  adamrt9319@gmail.com
"""

import pandas as pd
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from combine_model import GCN, CombinedModel, GradCAM
import dgl
import matplotlib.pyplot as plt
from utils import  set_random_seed
import pickle
from torch.utils.data import Dataset, DataLoader
from data import FC_Graph

set_random_seed(42)

# Load pretrained models
model1_state_dict = torch.load('saved_models_pth/gcn_model_1021_1.pth')
model2_state_dict = torch.load('saved_models_pth/gcn_model_single_graph_site_16_1028_1.pth')
# Initialize the models, assign the number
model1 = GCN(in_feats=379, hidden_feats=128, num_classes=379)
model2 = GCN(in_feats=128, hidden_feats=64, num_classes=128)

model1.load_state_dict(model1_state_dict)
model2.load_state_dict(model2_state_dict)

# Hyper-parameter
# FEATURE_DIM = 128
FEATURE_DIM = model2_state_dict['conv2.bias'].shape[0]
NUM_CLASSES = 2
LAMBDA_L1 = 1e-4
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

combined_model = CombinedModel(model1, model2,
                               feature_dim=FEATURE_DIM,
                               num_classes=NUM_CLASSES)

combined_model.to(DEVICE)

optimizer = Adam(combined_model.logistic.parameters(), lr=LR)
criterion = CrossEntropyLoss()


def l1_regularization(model, lambda_l1):
    """L1 regularization to encourage sparsity."""
    l1_norm = sum(p.abs().sum() for p in model.logistic.parameters())
    return lambda_l1 * l1_norm


def train(model, dataloader, optimizer, criterion, lambda_l1, device, num_epochs=5):
    """Training loop."""
    model.train()
    for epoch in range(num_epochs):
        for individual_data, g_pop, labels in dataloader:
            # Prepare individual data (list of tuples)
            individual_data = [(g.to(device), f.to(device)) for g, f in individual_data]
            g_pop = g_pop.to(device)

            # Convert labels to a tensor and move to device
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            outputs = model(individual_data, g_pop)
            loss = criterion(outputs, labels)
            # Add L1 regularization
            l1_loss = l1_regularization(model, lambda_l1)
            total_loss = loss + l1_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss.item()}")

# Step 5: Create a custom Dataset
class GraphDataset(Dataset):
    def __init__(self, graphs, subject_ids, labels, population_graph):
        self.graphs = graphs  # Dictionary {subject_id: graph}
        self.subject_ids = subject_ids  # List of subject IDs
        self.labels = labels  # List of labels corresponding to each subject
        self.population_graph = population_graph  # Population graph shared across all samples

    def __len__(self):
        # Return the total number of individual graphs (440 in this case)
        return len(self.subject_ids)

    def __getitem__(self, idx):
        # Retrieve the individual graph and its features based on the index
        subject_id = self.subject_ids[idx]
        individual_graph = self.graphs[subject_id]
        features = individual_graph.ndata['x']  # Adjust 'x' if the feature key is different

        # Retrieve the corresponding label for the individual graph
        label = self.labels[idx]

        # Return individual graph with its features, the population graph, and the label
        return (individual_graph, features), self.population_graph, label

def collate_fn(batch):
    # batch is a list of tuples: (individual_graph, population_graph, label)
    individual_graphs, population_graphs, labels = zip(*batch)

    # Combine individual graphs and their features into a list
    batched_individual_graphs = []
    for (graph, features) in individual_graphs:
        batched_individual_graphs.append((graph, features))

    # Since population graph is the same for each item, take the first one
    population_graph = population_graphs[0]

    # Convert labels to a single tensor
    labels = torch.tensor(labels)

    return batched_individual_graphs, population_graph, labels

if __name__ == "__main__":
    set_random_seed(42)
    # Step 1: Load g_subs and g_c3
    # Load g_subs from the pickle file
    with open('c1_graph_site_16/all_graphs_1.pkl', 'rb') as f:
        data = pickle.load(f)

    # Assume data is a dictionary with 'graphs' and 'subject_ids'
    # Dictionary {subject_id: graph}
    subject_ids = data.subject_ids
    graphs = data.graphs

    # Load g_c3
    with open('saved_pkl/g_c3_site_16_1021_1.pkl','rb') as f:
        g_c3_dataset=pickle.load(f)
        g_c3_node_ids=pickle.load(f)


    if isinstance(g_c3_dataset, list):
        g_c3_graph = g_c3_dataset[0]
    else:
        g_c3_graph = g_c3_dataset

    c3_subjectkeys = g_c3_node_ids
    y = g_c3_graph.ndata['y']

    # Create a mapping from subjectkey to label index
    subjectkey_to_label_index = {subjkey: idx for idx, subjkey in enumerate(c3_subjectkeys)}
    print('subjectkey_to label index: ',subjectkey_to_label_index)
    # Step 3: Transform y into binary labels
    y_binary = (y.sum(axis=1) > 0).int().tolist()  # Convert to binary labels: 1 if any of the 5 labels is positive, else 0

    # Create a mapping from subjectkey to binary label
    subjectkey_to_label = {subjkey: y_binary[idx] for subjkey, idx in subjectkey_to_label_index.items()}

    # Step 4: Match subjectkeys and prepare data
    # Find matching subject IDs
    matching_subject_ids = [sid for sid in subject_ids if sid in subjectkey_to_label]

    # Filter graphs and labels for matching subject IDs
    matched_graphs = {sid: graphs[sid] for sid in matching_subject_ids}
    matched_labels = [subjectkey_to_label[sid] for sid in matching_subject_ids]

    print('Matched Labels:', matched_labels)
    print('Matching IDs:', matching_subject_ids)
    print('Number of Matching IDs:', len(matching_subject_ids))

    # Instantiate the dataset
    dataset = GraphDataset(matched_graphs, matching_subject_ids, matched_labels, g_c3_graph)
    dataloader = DataLoader(
        dataset=GraphDataset(matched_graphs, matching_subject_ids, matched_labels, g_c3_graph),
        batch_size=440,  # Adjust batch size as needed
        collate_fn=collate_fn,
        shuffle=False
    )

    train(combined_model, dataloader, optimizer, criterion, lambda_l1=LAMBDA_L1, device=DEVICE,num_epochs=30)

    # YOUR MODFICATION HERE, save the model
    for param in combined_model.parameters():
        param.requires_grad = True

    # Use the input layer as the Grad-CAM target
    # Initialize Grad-CAM using the first graph convolution layer of `model1`
    target_layer = combined_model.model1.conv1

    # Initialize Grad-CAM with the target layer
    grad_cam = GradCAM(combined_model, target_layer)


    # Generate the heatmap using Grad-CAM
    for individual_data, g_pop, labels in dataloader:
        # Prepare individual data (list of tuples)
        individual_data = [(g.to(DEVICE), f.to(DEVICE)) for g, f in individual_data]
        g_pop = g_pop.to(DEVICE)
        heatmap = grad_cam(individual_data, g_pop)

    # Visualize the heatmap
    node_importance = heatmap
    print(heatmap)
    plt.bar(range(len(node_importance)), node_importance)  # Bar plot for node importance
    plt.xlabel("Node Index")
    plt.ylabel("Importance Score")
    plt.title("Node Importance Visualization")
    plt.show()
    print(type(heatmap))
    # df_heatmap = pd.DataFrame({
    #     "ROI": range(1, 380),  # ROI from 1 to 379
    #     "Value": heatmap
    # })
    # df_heatmap.to_csv('Node_importance.csv',index=False)




