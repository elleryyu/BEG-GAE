import os
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from data import ABCD_Subject2Subject
from model import GCN  
from utils import set_random_seed, visualize_embeddings,perform_logistic_regression,prepare_data_for_comparison,perform_weighted_logistic_regression
from sklearn.metrics import calinski_harabasz_score
from imblearn.over_sampling import SMOTE



def embedding_generation(model, device, graph):
    model.eval()

    with torch.no_grad():
        graph = graph.to(device)


        _, embeddings = model(graph, graph.ndata['feature'])
        embeddings=embeddings.detach().cpu().numpy()

    return embeddings

if __name__ == '__main__':
    set_random_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    with open('train/flattened_fc_graph.pkl', 'rb') as f:
        dataset = pickle.load(f)


    
    graph = dataset[0] 

    
    in_feats = graph.ndata['feature'].shape[1]
    y = graph.ndata['label']  
    print(f"Original y shape: {y.shape}")

    
    y_binary = torch.sum(y, dim=1)
    y_binary = (y_binary >= 1).long()  
    print(f"Converted binary y: {y_binary}")

    
    hidden_feats = 256
    num_classes = in_feats  
    model = GCN(in_feats=in_feats, hidden_feats=hidden_feats, num_classes=num_classes).to(device)

    # model_path = 'train/saved_models/gae_model_flattened_fc_1021.pth'
    model_path = 'train/saved_models/gae_model_flattened_fc_1021.pth'
    
    try:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {model_path}.")
    except FileNotFoundError:
        print(f"Model weights file {model_path} not found. Please check the file path.")

    embeddings = embedding_generation(model, device, graph)
    visualize_embeddings(embeddings, y=y, perplexity=15, radius= 0.25,filename='result_0303/site_16/tsne_gae_site16_fc.png')

    y_internalizing = (y[:, 0] + y[:, 1] >= 1).long()  # Internalizing
    y_externalizing = (y[:, 2] + y[:, 3] + y[:, 4] >= 1).long()  # Externalizing
    y_HC = (torch.sum(y, dim=1) == 0).long()  # Healthy controls

    # Internalizing vs HC
    X_internalizing_HC, y_internalizing_HC = prepare_data_for_comparison(embeddings, y_internalizing, y_HC)
    print("\nLogistic Regression: Internalizing vs HC")
    perform_logistic_regression(X_internalizing_HC, y_internalizing_HC,save_path='result_0303/site_16/gae_measure_internalizing_HC.csv')

    # Externalizing vs HC
    X_externalizing_HC, y_externalizing_HC = prepare_data_for_comparison(embeddings, y_externalizing, y_HC)
    print("\nLogistic Regression: Externalizing vs HC")
    perform_logistic_regression(X_externalizing_HC,y_externalizing_HC,save_path='result_0303/site_16/gae_measure_externalizing_HC.csv')

    # # Internalizing vs Externalizing
    # X_internalizing_externalizing, y_internalizing_externalizing = prepare_data_for_comparison(embeddings,
    #                                                                                            y_internalizing,
    #                                                                                            y_externalizing)
    # print("\nLogistic Regression: Internalizing vs Externalizing")
    # perform_logistic_regression(X_internalizing_externalizing, y_internalizing_externalizing)

    y_DX = (torch.sum(y, dim=1) > 0).long()

    # HC vs DX
    X_HC_DX, y_HC_DX = prepare_data_for_comparison(embeddings, y_HC, y_DX)
    print("\nLogistic Regression: HC vs DX")
    perform_logistic_regression(X_HC_DX, y_HC_DX,save_path='result_0303/site_16/gae_measure.csv')
    
    ch_score_DX = calinski_harabasz_score(X_HC_DX, y_HC_DX)
    print(f"Calinski-Harabasz Score (DX): {ch_score_DX:.2f}")


    smote=SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_HC_DX, y_HC_DX)
    print(len(X_resampled))

    # print("\nLogistic Regression: HC vs DX After SMOTE")
    # perform_logistic_regression(X_resampled, y_resampled,save_path='result_0303/site_21/trained_result/gae_measure_SMOTE.csv')

    # print("\n Weighted Logistic Regression: HC vs DX ")
    # perform_weighted_logistic_regression(X_HC_DX, y_HC_DX, save_path='result_0303/site_21/trained_result/fc_measure_Weighted.csv')


    ch_score_smote = calinski_harabasz_score(X_resampled, y_resampled)
    print(f"Calinski-Harabasz Score after SMOTE: {ch_score_smote:.2f}")