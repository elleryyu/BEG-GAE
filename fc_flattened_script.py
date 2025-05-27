import numpy as np
import pandas as pd

# Fisher Z transformation 
# def z_transform(r, n):
#     r = np.clip(r, -0.999, 0.999)
#     return np.arctanh(r) * np.sqrt(n - 3)

if __name__ == "__main__":
    # Define path here
    flattened_fc_path='site_16/site_16_flattened_fc.csv'
    labels_path='site_16/site_16_labels.csv'
    saved_path='fc_and_labels_transformed.csv'

    
    flattened_fc = pd.read_csv(flattened_fc_path, index_col=0)
    labels = pd.read_csv(labels_path, index_col=0)


    flattened_fc.index = flattened_fc.index.astype(str).str.strip()
    labels.index = labels.index.astype(str).str.strip()

    # evaluate index
    common_indices = flattened_fc.index.intersection(labels.index)
    print(f"Common indices count: {len(common_indices)}")
    print(f"Flattened FC only indices: {len(flattened_fc.index.difference(labels.index))}")
    print(f"Labels only indices: {len(labels.index.difference(flattened_fc.index))}")

    # Merge FC and labels
    merged_df = flattened_fc.join(labels, how='inner')  

    print(f"Merged DataFrame shape: {merged_df.shape}")
    print(merged_df.head())


    fc_cols = flattened_fc.columns
    label_cols = labels.columns

    X = merged_df[fc_cols].values
    y = merged_df[label_cols].values


    n = 379
    X_scaled = z_transform(X, n)
    fc_transformed_df = pd.DataFrame(X_scaled, columns=fc_cols, index=merged_df.index)
    final_df = pd.concat([fc_transformed_df, merged_df[label_cols]], axis=1)

    # Save as new csv
    final_df.to_csv(saved_path)
