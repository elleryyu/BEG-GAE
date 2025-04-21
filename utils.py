import random

import numpy as np
import torch
import dgl
import torch.nn.functional as F
from matplotlib.patches import Wedge
from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, auc
from sklearn.utils import resample
from sklearn.manifold import TSNE
import os
import random
import dgl
import umap

# Set random seed


def set_random_seed(seed=42):
    os.environ["PYTHONSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
    np.random.seed(seed)
    random.seed()
    dgl.seed(seed)             # DGL-specific seed
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def z_transform(r, n):

    if r >= 1.0:
        r = 0.999
    elif r <= -1.0:
        r = -0.999
    return np.log((1 + r) / (1 - r)) * (np.sqrt(n - 3) / 2)

def collate_func(samples):

    subs,graphs=map(list,zip(*samples))
    g_s=dgl.batch(graphs)
    return g_s,subs

def compute_euclidean_distance(X):
    num_nodes = X.shape[0]
    D = torch.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(num_nodes):
            D[i, j] = torch.norm(X[i] - X[j], p=2)

    return D

def compute_edges(X, W):
    # X is featue matrix, and W is learnable parameter
    # e_ij = X_i^T W X_j
    edge_weights = torch.matmul(X, W @ X.T)
    threshold = 0.5
    adj_matrix = (edge_weights > threshold).float()

    return adj_matrix, edge_weights

def generate_positive_negative_pairs(num_nodes):
    pos_pairs = [(i, i + 1) for i in range(num_nodes - 1)]
    neg_pairs = [(i, np.random.randint(0, num_nodes - 1)) for i in range(num_nodes)]
    return pos_pairs, neg_pairs

def loss_fn(edge_logits, reconstructed_h_u, g):
    h_u = g.ndata['h'][g.edges()[0]]  # [E, F]
    recon_loss = F.mse_loss(reconstructed_h_u, h_u)

    return recon_loss

def custom_loss(E, X,alpha=0.1):
    # Minimize the features difference
    D = torch.diag(torch.sum(E, dim=1))
    L = D - E  #
    # when calculate smooth loss, it is possible to get zero
    smoothness_loss = torch.trace(torch.matmul(X.t(), torch.matmul(L, X)))

    sparsity_loss = torch.norm(E, p=2)

    loss = smoothness_loss + alpha * sparsity_loss

    print("smooth loss:", smoothness_loss)
    return loss



def visualize_embeddings(embeddings, y, perplexity=20, n_iter=3000, learning_rate=300, filename='tsne_plot.png'):
    """
    Visualize node embeddings using t-SNE dimensionality reduction and save the plot locally.

    Parameters:
    embeddings: Node embeddings (numpy array or torch.Tensor)
    y: Label array (numpy array or torch.Tensor)
    filename: The name of the file to save the plot (default: 'tsne_plot.png')
    """
    # Mapping from label indices to disease names
    disease_mapping = {
        0: "Anxiety (Anx)",
        1: "Obsessive-Compulsive Disorder (OCD)",
        2: "Attention-Deficit/Hyperactivity Disorder (ADHD)",
        3: "Oppositional Defiant Disorder (ODD)",
        4: "Conduct Disorder (Cond)",
        5:"Specific Phobia (Anx Subtype)"
    }

    # Define corresponding colors for each disease
    colors = ['#1C9DFC', '#572BB6','#FE0303', '#F5510A', '#E8F60C','#4E4E49']

    # Ensure embeddings and labels are numpy arrays
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity,
        n_iter=n_iter,learning_rate=learning_rate
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    # Scale embeddings to enlarge coordinate values
    embeddings_2d *= 1  # Adjust scaling factor as needed
    embeddings_2d[:, 0] *=2   # Scale x-coordinates by 1.5

    # Create the plot with square layout
    fig, ax = plt.subplots(figsize=(8, 8))  # Set the figure size to be square

    # Set a fixed radius to ensure all points have the same size
    radius = 0.5 # Adjust point size as needed

    for i in range(len(embeddings_2d)):
        x_coord, y_coord = embeddings_2d[i]
        labels = y[i]  # Label array of length 6

        # Get indices of labels that are 1
        active_labels = np.where(labels == 1)[0]

        # Add condition to prioritize label 5 over label 0
        if 0 in active_labels and 5 in active_labels:
            active_labels = np.array([5])  # Only keep label 5

        if len(active_labels) == 0:
            # No labels are active, plot as 'No Disease' (black)
            wedge = Wedge(center=(x_coord, y_coord), r=radius, theta1=0, theta2=360, color='#CCDEDE')
            ax.add_patch(wedge)
        elif len(active_labels) == 1:
            # Only one label is active, plot a circle with the corresponding color
            color = colors[active_labels[0]]
            wedge = Wedge(center=(x_coord, y_coord), r=radius, theta1=0, theta2=360, color=color)
            ax.add_patch(wedge)
        else:
            # Multiple labels are active, plot a pie chart
            num_active = len(active_labels)
            angle_per_label = 360 / num_active
            start_angle = 0

            for idx in active_labels:
                end_angle = start_angle + angle_per_label
                wedge = Wedge(
                    center=(x_coord, y_coord),
                    r=radius,
                    theta1=start_angle,
                    theta2=end_angle,
                    color=colors[idx]
                )
                ax.add_patch(wedge)
                start_angle = end_angle


    # Remove axis ticks and labels but keep the axis lines
    ax.set_xticks([])
    ax.set_yticks([])

    # Remove grid
    plt.grid(False)

    # Get the min and max for x and y without buffer
    x_min, x_max = embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max()
    y_min, y_max = embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max()

    # Find the larger range and set both x and y limits to make the plot square
    range_max = max(1.1*(x_max - x_min), 1.1*(y_max - y_min))

    # Center the x and y limits around the data midpoints
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2

    # Set the x and y limits to make the plot square
    ax.set_xlim(x_mid - range_max / 2, x_mid + range_max / 2)
    ax.set_ylim(y_mid - range_max / 2, y_mid + range_max / 2)

    # Ensure the plot has a square aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Customize axis lines to ensure they are visible and form a square
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    # Set line widths for the axes (optional for visibility)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    # Save the plot to a file
    plt.savefig(filename)

    # Show the plot
    plt.show()




def check_sparsity(matrix):
    # Check instance
    if isinstance(matrix, torch.Tensor):
        total_elements = matrix.numel() 
        zero_elements = torch.sum(matrix == 1).item()  
    elif isinstance(matrix, np.ndarray):
        total_elements = matrix.size  
        zero_elements = np.sum(matrix == 0)  
    else:
        raise TypeError("Neither PyTorch Tensor nor NumPy ")

    sparsity = zero_elements / total_elements
    return sparsity

def prepare_data_for_comparison(X, y_group1, y_group2):
    """
    Prepare data for logistic regression comparison between two groups.

    Parameters:
    X: Embeddings (numpy array or torch.Tensor)
    y_group1: Binary labels for group 1 (torch.Tensor)
    y_group2: Binary labels for group 2 (torch.Tensor)

    Returns:
    X_filtered: Filtered embeddings for both groups
    y_filtered: Binary labels for the combined groups
    """
    # Select indices for group1 and group2
    indices_group1 = (y_group1 == 1).nonzero(as_tuple=True)[0]
    indices_group2 = (y_group2 == 1).nonzero(as_tuple=True)[0]

    # Filter embeddings for the selected indices
    X_filtered = X[torch.cat((indices_group1, indices_group2))]

    # Create labels: group1 as 1, group2 as 0
    y_filtered = torch.cat((torch.ones(len(indices_group1)), torch.zeros(len(indices_group2)))).numpy()

    return X_filtered, y_filtered

def perform_logistic_regression(X, y):
    """
    Perform logistic regression with balanced data and print the evaluation metrics.

    Parameters:
    X: Embeddings (numpy array)
    y: Labels (numpy array)
    """
    # Get all samples with y=1
    X_1 = X[y == 1]
    y_1 = y[y == 1]

    # Get all samples with y=0
    X_0 = X[y == 0]
    y_0 = y[y == 0]

    # Downsample the majority class to match the minority class size
    n_samples = min(len(X_1), len(X_0))  # Choose the smaller size to avoid errors

    if len(X_0) > len(X_1):
        # Downsample class 0
        X_0_downsampled, y_0_downsampled = resample(X_0, y_0,
                                                    replace=False,  # No replacement
                                                    n_samples=n_samples,  # Match the smaller number of samples
                                                    random_state=42)
        # Combine the downsampled class 0 data with class 1 data
        X_balanced = np.vstack((X_0_downsampled, X_1))
        y_balanced = np.hstack((y_0_downsampled, y_1))
    else:
        # Downsample class 1
        X_1_downsampled, y_1_downsampled = resample(X_1, y_1,
                                                    replace=False,  # No replacement
                                                    n_samples=n_samples,  # Match the smaller number of samples
                                                    random_state=42)
        # Combine the downsampled class 1 data with class 0 data
        X_balanced = np.vstack((X_0, X_1_downsampled))
        y_balanced = np.hstack((y_0, y_1_downsampled))

    # k fold cross-validation
    clf = LogisticRegression(random_state=42, max_iter=1000,penalty='l1',solver='liblinear')
    cv = StratifiedKFold(n_splits=5)

    accuracies = []
    f1_scores = []
    recalls = []
    aucs = []
    splits = list(cv.split(X_balanced, y_balanced))

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        X_train, X_test = X_balanced[train_idx], X_balanced[test_idx]
        y_train, y_test = y_balanced[train_idx], y_balanced[test_idx]

        # Normalize the embeddings
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the logistic regression model
        clf.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = clf.predict(X_test_scaled)
        y_pred_prob = clf.predict_proba(X_test_scaled)[:, 1]

        # Evaluate the model
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        aucs.append(auc(fpr, tpr))

    # Calculate mean and standard deviation for metrics
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    
    print(f"Mean accuracy: {mean_accuracy:.2f}")
    print(f"Standard deviation of accuracy: {std_accuracy:.2f}")

    print(f"Mean F1-score: {mean_f1:.2f}")
    print(f"Standard deviation of F1-score: {std_f1:.2f}")

    print(f"Mean recall: {mean_recall:.2f}")
    print(f"Standard deviation of recall: {std_recall:.2f}")

    print(f"Mean AUC: {mean_auc:.2f}")
    print(f"Standard deviation of AUC: {std_auc:.2f}")