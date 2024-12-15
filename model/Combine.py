import sys

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.optim as optim


class CombinedModel(torch.nn.Module):
    def __init__(self, GNN, Transformer):
        super(CombinedModel, self).__init__()
        self.GNN = GNN
        self.Transformer = Transformer
    def forward(self, H_v, H_e, H_e_pad, adj_e, adj_v, T, mask):
        # 通过第一个模型
        output = self.GNN(H_v, H_e, adj_e, adj_v, T)
        pad_output = torch.zeros((H_e_pad.shape[0], H_e_pad.shape[1], H_e_pad.shape[2])).to('cuda')
        for i in range(H_e_pad.shape[0]):
            length = output[i].shape[0]
            pad_output[i, :length, :] = output[i]
        # 通过第二个模型
        output = self.Transformer(H_e_pad, pad_output, mask)
        return output