import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import GraphConvolution

class CensNet(nn.Module):
    def __init__(self, nfeat, dropout):
        super(CensNet, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nfeat, nfeat, nfeat, node_layer=True)
        self.gc2 = GraphConvolution(nfeat, nfeat, nfeat, nfeat, node_layer=False)
        self.dropout = dropout

    def forward(self, X_list, Z_list, adj_e_list, adj_v_list, T_list):
        batch_size = len(T_list)
        outputs = []
        for i in range(batch_size):
            X = X_list[i].float()
            Z = Z_list[i].float()
            adj_e = adj_e_list[i].float()
            adj_v = adj_v_list[i].float()
            T = T_list[i].float()
            gc1 = self.gc1(X, Z, adj_e, adj_v, T)
            X, Z = F.relu(gc1[0]), F.relu(gc1[1])
            X = F.dropout(X, self.dropout, training=self.training)
            Z = F.dropout(Z, self.dropout, training=self.training)
            gc2 = self.gc2(X, Z, adj_e, adj_v, T)
            X, Z = F.relu(gc2[0]), F.relu(gc2[1])
            X = F.dropout(X, self.dropout, training=self.training)
            Z = F.dropout(Z, self.dropout, training=self.training)
            outputs.append(Z)
        return outputs
