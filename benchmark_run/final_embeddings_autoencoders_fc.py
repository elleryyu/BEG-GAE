import os
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import numpy as np
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from model import AutoEncoder 
from utils import set_random_seed, visualize_embeddings
import pandas  as pd
from utils import perform_logistic_regression,prepare_data_for_comparison,set_random_seed,perform_weighted_logistic_regression
from sklearn.metrics import calinski_harabasz_score
from imblearn.over_sampling import SMOTE

def embedding_generation(model, device, X_tensor):
    model.eval()
    with torch.no_grad():
        X_tensor = X_tensor.to(device)
        _,embeddings= model(X_tensor)  
        embeddings = embeddings.detach().cpu().numpy()
    return embeddings
if __name__ == '__main__':
    set_random_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #'site_16/fc_and_labels_transformed.csv'


    data = pd.read_csv('site_16/fc_and_labels_transformed.csv', index_col=0)

    label_cols = ['Anx', 'OCD', 'ADHD', 'ODD', 'Cond']
    label_cols = [col for col in label_cols if col in data.columns]


    fc_cols = data.columns.difference(label_cols)
    X = data[fc_cols].values
    y = data[label_cols].values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)  # Convert y to a PyTorch tensor

    input_dim = X_tensor.shape[1]
    hidden_dim = 1024  
    latent_dim = 256
    model = AutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

    # model_path = 'train/saved_models/autoencoder_best.pth'
    model_path='train/saved_models/autoencoder_best_site_21.pth'
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model weights from {model_path}.")
    except FileNotFoundError:
        print(f"Model weights file {model_path} not found. Please check the file path.")


    embeddings = embedding_generation(model, device, X_tensor)

    # 创建标签 using y_tensor instead of y
    y_internalizing = (y_tensor[:, 0] + y_tensor[:, 1] >= 1).long()  # Internalizing
    y_externalizing = (y_tensor[:, 2] + y_tensor[:, 3] + y_tensor[:, 4] >= 1).long()  # Externalizing
    y_HC = (torch.sum(y_tensor, dim=1) == 0).long()  # Healthy controls

    # Internalizing vs HC
    X_internalizing_HC, y_internalizing_HC = prepare_data_for_comparison(embeddings, y_internalizing, y_HC)
    print("\nLogistic Regression: Internalizing vs HC")
    perform_logistic_regression(X_internalizing_HC, y_internalizing_HC,save_path='result_0303/site_16/test_result/ae_measure_internalizing_HC.csv')

    # Externalizing vs HC
    X_externalizing_HC, y_externalizing_HC = prepare_data_for_comparison(embeddings, y_externalizing, y_HC)
    print("\nLogistic Regression: Externalizing vs HC")
    perform_logistic_regression(X_externalizing_HC, y_externalizing_HC,save_path='result_0303/site_21/trained_result/ae_measure_externalizing_HC.csv')

    # # Internalizing vs Externalizing
    # X_internalizing_externalizing, y_internalizing_externalizing = prepare_data_for_comparison(embeddings, y_internalizing, y_externalizing)
    # print("\nLogistic Regression: Internalizing vs Externalizing")
    # perform_logistic_regression(X_internalizing_externalizing, y_internalizing_externalizing,saved_path='ae_measure_externalizing_HC')

    y_DX = (torch.sum(y_tensor, dim=1) > 0).long()

    # HC vs DX
    X_HC_DX, y_HC_DX = prepare_data_for_comparison(embeddings, y_HC, y_DX)
    print("\nLogistic Regression: HC vs DX")
    perform_logistic_regression(X_HC_DX, y_HC_DX,save_path='result_0303/site_21/trained_result/ae_measure_fc.csv')
    
    
    visualize_embeddings(embeddings,y,perplexity=15,radius=0.2,filename='result_0303/site_21/trained_result/tsne_autoencoder_fc.png')

    print('HC People', y_HC.sum())
    print('DX People', y_DX.sum())

    ch_score_DX = calinski_harabasz_score(X_HC_DX, y_HC_DX)
    print(f"Calinski-Harabasz Score (DX): {ch_score_DX:.2f}")


    smote=SMOTE(random_state=42)
    # X_resampled, y_resampled = smote.fit_resample(X_HC_DX, y_HC_DX)
    # print(len(X_resampled))

    # print("\nLogistic Regression: HC vs DX After SMOTE")
    # perform_logistic_regression(X_resampled, y_resampled,save_path='result_0303/ae_measure_SMOTE.csv')

    # print("\n Weighted Logistic Regression: HC vs DX ")
    # perform_weighted_logistic_regression(X_HC_DX, y_HC_DX, save_path='result_0303/fc_measure_Weighted.csv')


    ch_score_smote = calinski_harabasz_score(X_resampled, y_resampled)
    print(f"Calinski-Harabasz Score after SMOTE: {ch_score_smote:.2f}")