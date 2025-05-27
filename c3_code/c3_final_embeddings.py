import os
import torch
from sklearn.cluster import KMeans
import pickle
from model import GCN
from utils import set_random_seed,visualize_embeddings,perform_logistic_regression,prepare_data_for_comparison

# ==== Configuration ====
GRAPH_PATH = 'saved_pkl/g_c3_site_16_1028_max.pkl'
MODEL_PATH = 'saved_models_pth/gcn_model_single_graph_site_16_1028_1.pth'
TSNE_FILENAME = 'tsne_pro_method_max_1120_2yrFL_with_sph.png'
HIDDEN_FEATS = 64
SEED = 42


# ==== Embedding Generation ====
def embedding_generation(model, device, graph):
    model.eval()
    with torch.no_grad():
        graph = graph.to(device)
        _, embeddings = model(graph, graph.ndata['x'])
        return embeddings.detach().cpu().numpy()


# ==== Main ====
def main():
    # Set seed and device
    set_random_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load graph
    with open(GRAPH_PATH, 'rb') as f:
        dataset = pickle.load(f)

    graph = dataset[0] if isinstance(dataset, list) else dataset
    in_feats = graph.ndata['x'].shape[1]
    y = graph.ndata['y']
    print(f"Original label matrix shape: {y.shape}")

    # Initialize model
    model = GCN(in_feats=in_feats, hidden_feats=HIDDEN_FEATS, num_classes=in_feats).to(device)

    # Load model weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded model weights from: {MODEL_PATH}")
    else:
        print(f"Model weights file not found at {MODEL_PATH}. Exiting.")
        return

    # Generate embeddings
    embeddings = embedding_generation(model, device, graph)

    # Visualize using t-SNE
    visualize_embeddings(embeddings, y=y, filename=TSNE_FILENAME)

    # Derive diagnostic labels
    y_internalizing = (y[:, 0] + y[:, 1] >= 1).long()
    y_externalizing = (y[:, 2] + y[:, 3] + y[:, 4] >= 1).long()
    y_HC = (torch.sum(y, dim=1) == 0).long()
    y_DX = (torch.sum(y, dim=1) > 0).long()

    # Logistic Regression Classifications
    classification_tasks = [
        ("Internalizing vs HC", y_internalizing, y_HC),
        ("Externalizing vs HC", y_externalizing, y_HC),
        ("Internalizing vs Externalizing", y_internalizing, y_externalizing),
        ("HC vs DX", y_HC, y_DX),
    ]

    for task_name, y1, y2 in classification_tasks:
        X_task, y_task = prepare_data_for_comparison(embeddings, y1, y2)
        print(f"\nLogistic Regression: {task_name}")
        perform_logistic_regression(X_task, y_task)


if __name__ == '__main__':
    main()