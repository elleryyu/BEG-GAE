import os
import torch
from sklearn.cluster import KMeans
import pickle
from model import GCN
from utils import set_random_seed,visualize_embeddings,perform_logistic_regression,prepare_data_for_comparison

def embedding_generation(model, device, graph):
    model.eval()

    with torch.no_grad():
        graph = graph.to(device)


        _, embeddings = model(graph, graph.ndata['x'])
        embeddings=embeddings.detach().cpu().numpy()

    return embeddings

if __name__ == '__main__':
    set_random_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    with open('saved_pkl/g_c3_site_16_1028_max.pkl', 'rb') as f:
        dataset = pickle.load(f)

    if isinstance(dataset, list):
        graph = dataset[0]
    else:
        graph = dataset

    in_feats = graph.ndata['x'].shape[1]
    y = graph.ndata['y']  # Assuming y is a 5-column label matrix
    print(f"Original y shape: {y.shape}")

    hidden_feats = 64
    num_classes = in_feats

    model = GCN(in_feats=in_feats, hidden_feats=hidden_feats, num_classes=num_classes).to(device)

    model_path = 'saved_models_pth/gcn_model_single_graph_site_16_1028_1.pth'
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}.")
    except FileNotFoundError:
        print(f"Model weights file {model_path} not found. Please check the file path.")

    # Generate embeddings from the GCN model
    embeddings = embedding_generation(model, device, graph)


    visualize_embeddings(embeddings,y=y,filename='tsne_pro_method_max_1120_2yrFL_with_sph.png')


    y_internalizing = (y[:, 0] + y[:, 1] >= 1).long()  # Internalizing
    y_externalizing = (y[:, 2] + y[:, 3] + y[:, 4] >= 1).long()  # Externalizing
    y_HC = (torch.sum(y, dim=1) == 0).long()  # Healthy controls

    # Internalizing vs HC
    X_internalizing_HC, y_internalizing_HC = prepare_data_for_comparison(embeddings, y_internalizing, y_HC)
    print("\nLogistic Regression: Internalizing vs HC")
    perform_logistic_regression(X_internalizing_HC, y_internalizing_HC)

    # Externalizing vs HC
    X_externalizing_HC, y_externalizing_HC = prepare_data_for_comparison(embeddings, y_externalizing, y_HC)
    print("\nLogistic Regression: Externalizing vs HC")
    perform_logistic_regression(X_externalizing_HC, y_externalizing_HC)

    # Internalizing vs Externalizing
    X_internalizing_externalizing, y_internalizing_externalizing = prepare_data_for_comparison(embeddings, y_internalizing, y_externalizing)
    print("\nLogistic Regression: Internalizing vs Externalizing")
    perform_logistic_regression(X_internalizing_externalizing, y_internalizing_externalizing)

    y_DX = (torch.sum(y, dim=1) > 0).long()

    # HC vs DX
    X_HC_DX, y_HC_DX = prepare_data_for_comparison(embeddings, y_HC, y_DX)
    print("\nLogistic Regression: HC vs DX")
    perform_logistic_regression(X_HC_DX, y_HC_DX)