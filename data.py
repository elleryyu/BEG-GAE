"""
Author: Weifeng Yu (Ellery)
Created:  '2024-09-01'
Last Modified:2024-04-26
Email: dag9wj@virginia.edu
"""

import os
import pandas as pd
import numpy as np
import torch
import pickle
from dgl.data import DGLDataset
import dgl
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from utils import z_transform  # Import the z_transform function from your utils.py


class FC_Graph(DGLDataset):

    def __init__(self, dataset_path=r"./data_copied", label_path=r"./labels.csv",
                  train_ratio=0.8, val_ratio=0.1, random_seed=20):

        self.dataset_path = dataset_path

        if dataset_path is not None:
            self.subject_ids, self.dataset = self.get_x_from_directory(dataset_path)
        else:
            raise ValueError('No data provided')

        # self.train_subject_ids, self.val_subject_ids, self.test_subject_ids = self.split_data(
        #     self.subject_ids, train_ratio, val_ratio, random_seed)
        self.train_ratio=train_ratio
        self.val_ratio=val_ratio
        self.graphs = self.process_data(self.dataset)

    def split_data(self, subject_ids, train_ratio, val_ratio, random_seed):
        test_ratio = 1 - train_ratio - val_ratio
        train_subject_ids, test_subject_ids = train_test_split(
            subject_ids, test_size=test_ratio, random_state=random_seed)
        train_subject_ids, val_subject_ids = train_test_split(
            train_subject_ids, test_size=val_ratio / (train_ratio + val_ratio), random_state=random_seed)

        return train_subject_ids, val_subject_ids, test_subject_ids

    def get_x_from_directory(self, dataset_path):
        all_files = os.listdir(dataset_path)
        subject_ids = list(set([f.split('.')[0] for f in all_files if '.csv' in f]))
        dfs = {}
        for subject_id in subject_ids:
            dfs[subject_id] = pd.read_csv(f'{dataset_path}/{subject_id}.csv')

        return subject_ids, dfs

    # Process the data and generate the adjacency matrix
    def process_data(self, dataset):
        graphs = {}
        for subject_id, data in dataset.items():
            features = data.values


            if np.isnan(features).any():
                raise ValueError(f"Error: There are NaN values in the features of subject {subject_id}.")


            preprocessed_features = self.preprocess_adjacency_matrix(features)
            adjacency_matrix = kneighbors_graph(preprocessed_features, n_neighbors=30, mode='connectivity',
                                                include_self=False)
            adjacency_matrix = adjacency_matrix.toarray()


            adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
            x_tensor = torch.tensor(features, dtype=torch.float32)


            src, dst = np.nonzero(adjacency_matrix)
            src = torch.tensor(src, dtype=torch.long)  # Source nodes
            dst = torch.tensor(dst, dtype=torch.long)  # Destination nodes


            g = dgl.graph((src, dst))
            g.ndata['x'] = x_tensor
            graphs[subject_id] = g

        return graphs

    # Preprocess the adjacency matrix using z_transform
    def preprocess_adjacency_matrix(self, matrix):
        n_val = 379  # Total number of elements in the matrix
        print('n_val is:',n_val)
        # Apply z_transform to each element of the matrix using np.vectorize
        z_transform_func = np.vectorize(lambda r: z_transform(r, n_val))
        transformed_matrix = z_transform_func(matrix)

        # Set the diagonal to 0 to prevent self-connections
        np.fill_diagonal(transformed_matrix, 0)
        return transformed_matrix

    # Generate the graph structure
    def __getitem__(self, idx):
        subject_id=self.subject_ids[idx]

        return subject_id,self.graphs[subject_id]


    def __len__(self):

        return len(self.graphs)


class Graph_with_adjacency_matrix(DGLDataset):
    def __init__(self, adjacency_matrix,embedding_path=r"./graph_embeddings.pkl",label_path='./ABCD_site_11_6labels.csv'):

        self.embedding_path = embedding_path
        self.label_path = label_path

        self.adjacency_matrix=adjacency_matrix

        if embedding_path is not None:
            self.subject_ids, self.embeddings = self.get_x_from_embeddings_pickle(embedding_path)
        else:
            raise ValueError('No data provided')

        self.g = self.process_data()


    def get_x_from_embeddings_pickle(self,embedding_path):
        # read pickle
        with open(embedding_path, 'rb') as f:
            embeddings_dict = pickle.load(f)

        # get keys from list
        subject_ids = list(embeddings_dict.keys())

        # save
        dfs = {}
        for subject_id in subject_ids:
            dfs[subject_id] = embeddings_dict[subject_id]
        return subject_ids, dfs


    # Process the data and generate the adjacency matrix
    def process_data(self):

        ########################################################
        # Obtain and check indices of embeddings and adj_matrix#
        ########################################################

        embedding_indices = list(self.embeddings.keys())
        adjacency_indices = list(self.adjacency_matrix.index)
        print('emb_ind:', embedding_indices)
        print('adj_ind: ',adjacency_indices)
        if embedding_indices != adjacency_indices:
            if isinstance(self.embeddings, dict):
                print("Reordering embeddings to match adjacency matrix order.")

                ordered_embeddings = [self.embeddings[idx] for idx in adjacency_indices]

            else:
                raise TypeError("self.embeddings should be a dictionary.")

        ##################################################################
        features = np.array(ordered_embeddings)

        if np.isnan(features).any():
            raise ValueError(f"Error: There are NaN values in the features .")

        ##########################################################
        # Read labels and add them to the graph
        ##########################################################
        labels_df = pd.read_csv(self.label_path)


        label_list = []
        for node_id in adjacency_indices:

            matching_row = labels_df[labels_df.iloc[:, 0] == node_id]
            if not matching_row.empty:
                label_list.append(matching_row.iloc[0, 1:].values)
            else:
                print('Can not find node id:',node_id)
                label_list.append([0] * 6)

        label_array = np.array(label_list,dtype=np.float32)

        adjacency_matrix_np = self.adjacency_matrix.values
        src, dst = np.nonzero(adjacency_matrix_np)

        g = dgl.graph((src, dst))

        x_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(label_array, dtype=torch.float32)

        g.ndata['x'] = x_tensor
        g.ndata['y'] = y_tensor
        self.node_ids = adjacency_indices

        return g

    def __getitem__(self, idx):
        if idx==0:
            return self.g, self.node_ids
        else:
            assert 'Here is only one graph, wrong prompt'
    def __len__(self):
        return 1


class ABCD_Subject2Subject(DGLDataset):
    def __init__(self, data_csv_path=r"./site.csv",
                 similarity_threshold=None, random_seed=42, dst_threshold=None):
        super().__init__(name='HCDP')

        self.data_csv_path = data_csv_path
        self.similarity_threshold = similarity_threshold
        self.dst_threshold = dst_threshold
        self.random_seed = random_seed

        # Read data and process
        self.node_features, self.labels, self.node_indices, self.adjacency_matrix = self.process_data()

    def process_data(self):
        # Read data CSV
        data_df = pd.read_csv(self.data_csv_path, index_col=0)
        data_df = data_df.dropna()  # Remove rows with NaN values

        # Define label columns
        label_columns = ['Anx', 'OCD', 'ADHD', 'ODD', 'Cond']
        feature_columns = [col for col in data_df.columns if col not in label_columns]

        # Extract features and labels
        node_features = data_df[feature_columns].values
        labels = data_df[label_columns].values

        # Create adjacency matrix using k-nearest neighbors
        adjacency_matrix = kneighbors_graph(node_features, n_neighbors=25, mode='connectivity', include_self=False)
        adjacency_matrix = adjacency_matrix.toarray()
        adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
        np.fill_diagonal(adjacency_matrix, 0)

        print(f"Adjacency matrix sparsity: {adjacency_matrix.sum() / adjacency_matrix.size:.4f}")

        node_indices = data_df.index.tolist()

        return node_features, labels, node_indices, adjacency_matrix

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("This dataset contains only one graph.")

        x_tensor = torch.tensor(self.node_features, dtype=torch.float32)

        # Create undirected edges
        src, dst = np.nonzero(np.triu(self.adjacency_matrix, k=1))
        edges = np.concatenate((np.vstack((src, dst)), np.vstack((dst, src))), axis=1)
        src, dst = edges

        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)

        g = dgl.graph((src, dst))
        g.ndata['feature'] = x_tensor
        g.ndata['label'] = torch.tensor(self.labels, dtype=torch.long)

        return g

    def __len__(self):
        return 1


class Graph_with_knn(DGLDataset):
    def __init__(self, embedding_path=r"./graph_embeddings.pkl", label_path='./ABCD_site_11_6labels.csv', k=25):
        self.embedding_path = embedding_path
        self.label_path=label_path
        self.label_path = label_path
        self.k = k

        if embedding_path is not None:
            self.subject_ids, self.embeddings = self.get_x_from_embeddings_pickle(embedding_path)
        else:
            raise ValueError('No data provided')

        self.g = self.process_data()

    def get_x_from_embeddings_pickle(self, embedding_path):
        with open(embedding_path, 'rb') as f:
            embeddings_dict = pickle.load(f)

        subject_ids = list(embeddings_dict.keys())

        dfs = {}
        for subject_id in subject_ids:
            dfs[subject_id] = embeddings_dict[subject_id]
        return subject_ids, dfs

    def process_data(self):

        embedding_indices = list(self.embeddings.keys())


        if isinstance(self.embeddings, dict):

            ordered_embeddings = np.array([self.embeddings[idx] for idx in embedding_indices])
        else:
            raise TypeError("self.embeddings should be a dictionary.")
        
        adjacency_matrix = kneighbors_graph(ordered_embeddings, self.k, mode='connectivity', include_self=False)

        # 确保邻接矩阵是对称的
        adjacency_matrix = adjacency_matrix + adjacency_matrix.T 
        adjacency_matrix[adjacency_matrix > 1] = 1  

        labels_df = pd.read_csv(self.label_path)

        label_list = []
        for node_id in embedding_indices:
            matching_row = labels_df[labels_df.iloc[:, 0] == node_id]
            if not matching_row.empty:
                label_list.append(matching_row.iloc[0, 1:].values)
            else:
                print('Can not find node id:', node_id)
                label_list.append([0] * 6)

        label_array = np.array(label_list, dtype=np.float32)

        adjacency_matrix = adjacency_matrix.toarray()  
        src, dst = np.nonzero(adjacency_matrix)
        g = dgl.graph((src, dst))

        x_tensor = torch.tensor(ordered_embeddings, dtype=torch.float32)
        y_tensor = torch.tensor(label_array, dtype=torch.float32)

        g.ndata['x'] = x_tensor
        g.ndata['y'] = y_tensor

        self.node_ids = embedding_indices
        return g

    def __getitem__(self, idx):
        if idx == 0:
            return self.g, self.node_ids
        else:
            raise IndexError('There is only one graph, wrong index.')

    def __len__(self):
        return 1


if __name__ == "__main__":

    save_path='c1_graph_site_16'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # similarity_threshold = 0.995
    train_ratio = 0.7
    val_ratio = 0.1
    random_seed = 2
    dataset = FC_Graph(dataset_path='./site_16/site_16_FC',
                       train_ratio=train_ratio, val_ratio=val_ratio, random_seed=random_seed)
    sub_ids= dataset.subject_ids
    graphs=dataset.graphs

    print(len(dataset))
    with open(f'{save_path1}/all_graphs_1.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    print("All graphs have been saved to './c1_graph_site_16/all_graphs_1.pkl'.")


