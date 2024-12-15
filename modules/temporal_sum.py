import torch
import torch_scatter as scatter
from torch import nn
import torch.nn.functional as F

from modules.utils import MergeLayer

class TemporalSumLayer(torch.nn.Module):
  """
  Temporal attention layer. Return the temporal embedding of a node given the node itself,
   its neighbors and the edge timestamps.
  """

  def __init__(self, n_node_features, n_edge_features, time_dim,
               output_dimension):
    super(TemporalSumLayer, self).__init__()
    self.time_dim = time_dim

    self.reverse_flag = True
    self.selfloop_flag = True

    self.query_dim = n_node_features + time_dim
    self.key_dim = n_node_features + time_dim + n_edge_features

    self.out_dim = output_dimension


    self.merger1 = torch.nn.Linear(n_node_features + time_dim + n_edge_features, self.out_dim)
    self.merger2 = torch.nn.Linear(n_node_features + time_dim + self.out_dim, self.out_dim)


  def forward(self, node_feature, edge_index, edge_feature, src_time_features, edge_time, mask=None, sample_ratio=None):
    '''
    无向图，边增加为双向
    增加自闭环
    :param node_feature:
    :param edge_index:
    :param edge_feature:
    :param src_time_features:
    :param edge_time:
    :param mask:
    :return:
    '''
    if mask is not None and sample_ratio is None:
      edge_index, edge_feature, src_time_features, edge_time = self.mask_edge(edge_index,
                                                                              edge_feature,
                                                                              src_time_features,
                                                                              edge_time,
                                                                              mask)

    if self.reverse_flag:
      edge_index, edge_feature, src_time_features, edge_time, sample_ratio = self.reverse_edge(edge_index,
                                                                                 edge_feature,
                                                                                 src_time_features,
                                                                                 edge_time, sample_ratio)
    if self.selfloop_flag:
      edge_index, edge_feature, src_time_features, edge_time, sample_ratio = self.add_selfloop(node_feature,
                                                                                 edge_index,
                                                                                 edge_feature,
                                                                                 src_time_features,
                                                                                 edge_time, sample_ratio)
    node_i = edge_index[:, 0]
    node_j = edge_index[:, 1]
    node_feat_i = node_feature[node_i, :]
    node_feat_j = node_feature[node_j, :]

    source_node_vec = torch.cat([node_feat_i, src_time_features], dim=1)
    source_node_vec = scatter.scatter_mean(source_node_vec, node_i, dim=0)
    target_node_vec = torch.cat([node_feat_j, edge_feature, edge_time], dim=1)

    neighbor_embeddings = self.merger1(target_node_vec)

    if sample_ratio is not None:
      neighbor_embeddings = torch.multiply(neighbor_embeddings, sample_ratio.reshape([-1, 1]))

    neighbors_sum = scatter.scatter_sum(neighbor_embeddings, node_i, dim=0)
    neighbors_sum = F.relu(neighbors_sum)

    source_embedding = torch.cat([neighbors_sum, source_node_vec], dim=1)

    out_emb = self.merger2(source_embedding)

    return out_emb


  def reverse_edge(self, edge_index, edge_feature, src_time_features, edge_time, sample_ratio):
    reverse_edge_index = torch.cat((edge_index[:, 1].unsqueeze(1), edge_index[:, 0].unsqueeze(1)), dim=1)
    two_edge_index = torch.cat((edge_index, reverse_edge_index), dim=0)
    src_time_features = src_time_features.repeat(2, 1)
    edge_feature = edge_feature.repeat(2, 1)
    edge_time = edge_time.repeat(2, 1)

    if sample_ratio is not None:
      sample_ratio = sample_ratio.repeat(2)

    return two_edge_index, edge_feature, src_time_features, edge_time, sample_ratio

  def add_selfloop(self, node_feature, edge_index, edge_feature, src_time_features, edge_time, sample_ratio):
    time_emb_unit = src_time_features[0, :].reshape(1, -1)
    node_id = torch.arange(0, node_feature.shape[0], device=edge_index.device).reshape(-1,1)
    edge_index = torch.cat([edge_index, node_id.repeat(1,2)], dim=0)
    edge_feature = torch.cat([edge_feature, torch.zeros([node_id.shape[0], edge_feature.shape[1]], dtype=edge_feature.dtype, device=edge_feature.device)], dim=0)
    src_time_features = torch.cat([src_time_features, time_emb_unit.repeat(node_id.shape[0], 1)], dim=0)
    edge_time = torch.cat([edge_time, time_emb_unit.repeat(node_id.shape[0], 1)], dim=0)


    if sample_ratio is not None:
      sample_ratio =torch.cat([sample_ratio, torch.ones([node_id.shape[0]], dtype=sample_ratio.dtype, device=sample_ratio.device)])

    return edge_index, edge_feature, src_time_features, edge_time, sample_ratio

  def mask_edge(self, edge_index, edge_feature, src_time_features, edge_time, mask):
    retain_index = torch.nonzero(mask).reshape([-1])
    edge_index = edge_index[retain_index]
    edge_feature = edge_feature[retain_index]
    src_time_features = src_time_features[retain_index]
    edge_time = edge_time[retain_index]

    return edge_index, edge_feature, src_time_features, edge_time

