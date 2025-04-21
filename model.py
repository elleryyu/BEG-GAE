"""
Author: Weifeng Yu (Ellery)
Created:  2024-09-01
Last Modified:2024-12-09
Email: dag9wj@virginia.edu
"""

import torch
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import  dgl
import pickle
import pandas as pd
from utils import compute_euclidean_distance, check_sparsity,set_random_seed,create_adj_with_ste,visualize_embeddings
import numpy as np


class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dglnn.GraphConv(hidden_feats, num_classes)

    def forward(self, g, features):
        #64->128->64
        x = self.conv1(g, features)
        h = x
        x = F.relu(x)
        x = self.conv2(g, x)
        return x,h



class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, latent_dim=256):
        super(AutoEncoder, self).__init__()
        # 
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 
            nn.ReLU(),  
            nn.Linear(hidden_dim, latent_dim),  #
            nn.ReLU()  
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class MLP(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(in_feats, hidden_feats)
        self.fc2 = torch.nn.Linear(hidden_feats, num_classes)

    def forward(self, features):
        h = F.relu(self.fc1(features))
        out = self.fc2(h)
        return out,h
