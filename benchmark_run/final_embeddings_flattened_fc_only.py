

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from utils import visualize_embeddings,perform_logistic_regression,set_random_seed

def prepare_data_for_comparison(X, y_group1, y_group2):
    # Select samples belonging to group1 or group2
    idx = (y_group1 == 1) | (y_group2 == 1)
    X_selected = X[idx]
    # Create labels: group1 as 1, group2 as 0
    y_selected = np.where(y_group1[idx] == 1, 1, 0)
    return X_selected, y_selected


set_random_seed()

file_path = '../site_16/fc_and_labels_transformed.csv'
merged_df = pd.read_csv(file_path, index_col=0)  # Load the merged file

# Assume the last 5 columns are labels, and the rest are flattened FC
fc_cols = merged_df.columns[:-5]  # All but the last 5 columns are flattened FC
label_cols = merged_df.columns[-5:]  # The last 5 columns are labels

# Extract flattened FC and labels
X = merged_df[fc_cols]  # Use flattened FC as features (kept as a DataFrame)
labels = merged_df[label_cols]  # Extract labels (also kept as a DataFrame)

print(f"Flattened FC shape: {X.shape}")
print(f"Labels shape: {labels.shape}")

# Step 2: Data standardization
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=fc_cols, index=X.index)  # Apply scaling and keep as DataFrame

# Create labels using Pandas operations
y_internalizing = (labels.iloc[:, 0] + labels.iloc[:, 1] >= 1).astype(int)  # Internalizing
y_externalizing = (labels.iloc[:, 2] + labels.iloc[:, 3] + labels.iloc[:, 4] >= 1).astype(int)  # Externalizing
y_HC = (labels.sum(axis=1) == 0).astype(int)


# Internalizing vs HC
X_internalizing_HC, y_internalizing_HC = prepare_data_for_comparison(X_scaled, y_internalizing, y_HC)
print("\nLogistic Regression: Internalizing vs HC")
perform_logistic_regression(X_internalizing_HC, y_internalizing_HC)

# Externalizing vs HC
X_externalizing_HC, y_externalizing_HC = prepare_data_for_comparison(X_scaled, y_externalizing, y_HC)
print("\nLogistic Regression: Externalizing vs HC")
perform_logistic_regression(X_externalizing_HC, y_externalizing_HC)

# Internalizing vs Externalizing
X_internalizing_externalizing, y_internalizing_externalizing = prepare_data_for_comparison(X_scaled, y_internalizing, y_externalizing)
print("\nLogistic Regression: Internalizing vs Externalizing")
perform_logistic_regression(X_internalizing_externalizing, y_internalizing_externalizing)

# Healthy controls vs any diagnosis (DX)
y_DX = (labels.sum(axis=1) > 0).astype(int)
X_HC_DX, y_HC_DX = prepare_data_for_comparison(X_scaled, y_HC, y_DX)
print("\nLogistic Regression: HC vs DX")
perform_logistic_regression(X_HC_DX, y_HC_DX)
visualize_embeddings(embeddings=X,y=labels.values,filename='tsne_flatten_fc.png')