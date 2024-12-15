import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import math
import time

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    #x = self.layer_norm(x)
    h = self.act(self.fc1(x))
    return self.fc2(h) + x2

class MergeLayer_output(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3= 1024, dim4=1, drop_out=0.2):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim3)
    self.fc3 = torch.nn.Linear(dim3, dim2)
    self.fc4 = torch.nn.Linear(dim2 , dim4 )
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop_out)

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x):
    h = self.act(self.fc1(x))
    h = self.act(self.fc2(h))
    h = self.dropout(self.act(self.fc3(h)))
    h = self.fc4(h)
    return h






class Feat_Process_Layer(torch.nn.Module):
  def __init__(self, dim1, dim2):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1, dim2)
    self.fc2 = torch.nn.Linear(dim2, dim2)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x):
    h = self.act(self.fc1(x))
    return self.fc2(h)


def drop_edge(src_neigh_edge, src_edge_to_time, src_org_edge_feat, ratio):
  num_edges = src_neigh_edge.shape[0]
  pickup_ids = torch.rand([num_edges]) > ratio

  src_neigh_edge = src_neigh_edge[pickup_ids]
  src_edge_to_time = src_edge_to_time[pickup_ids]
  src_org_edge_feat = src_org_edge_feat[pickup_ids]
  return src_neigh_edge,src_edge_to_time,src_org_edge_feat