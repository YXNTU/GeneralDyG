import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layers import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat_v, nfeat_e, nhid, nclass, dropout, node_layer=True):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat_v, nhid, nfeat_e, nfeat_e, node_layer=True)
        self.gc2 = GraphConvolution(nhid, nhid, nfeat_e, nfeat_e, node_layer=False)
        self.gc3 = GraphConvolution(nhid, nclass, nfeat_e, nfeat_e, node_layer=True)
        self.dropout = dropout

    def forward(self, X_list, Z_list, adj_e_list, adj_v_list, T_list):
        batch_size = len(T_list)
        outputs = []
        for i in range(batch_size):
            X = X_list[i]
            Z = Z_list[i]
            adj_e = adj_e_list[i]
            adj_v = adj_v_list[i]
            T = T_list[i]

            gc1 = self.gc1(X, Z, adj_e, adj_v, T)
            X, Z = F.relu(gc1[0]), F.relu(gc1[1])
            X = F.dropout(X, self.dropout, training=self.training)
            Z = F.dropout(Z, self.dropout, training=self.training)

            gc2 = self.gc2(X, Z, adj_e, adj_v, T)
            X, Z = F.relu(gc2[0]), F.relu(gc2[1])
            X = F.dropout(X, self.dropout, training=self.training)
            Z = F.dropout(Z, self.dropout, training=self.training)

            X, Z = self.gc3(X, Z, adj_e, adj_v, T)
            outputs.append(X)
        outputs = torch.stack(outputs, dim=0)
        return outputs
