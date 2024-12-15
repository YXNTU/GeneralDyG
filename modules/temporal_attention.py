import torch
import torch_scatter as scatter
from torch import nn

from modules.utils import MergeLayer

class TemporalAttentionLayer2(torch.nn.Module):
  """
  Temporal attention layer. Return the temporal embedding of a node given the node itself,
   its neighbors and the edge timestamps.
  """

  def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim,
               output_dimension, n_head=2, dropout=0.1):
    super(TemporalAttentionLayer2, self).__init__()
    self.time_dim = time_dim
    self.num_heads = n_head

    self.reverse_flag = True
    self.selfloop_flag = True

    self.query_dim = n_node_features + time_dim
    self.key_dim = n_node_features + time_dim + n_edge_features

    self.out_dim = output_dimension
    self.d_k = self.out_dim // self.num_heads
    self.scale = self.d_k ** (-0.5)

    self.q_linears = torch.nn.Sequential( torch.nn.Linear(self.query_dim, self.out_dim), torch.nn.ReLU())
    self.k_linears =  torch.nn.Sequential(torch.nn.Linear(self.key_dim, self.out_dim), torch.nn.ReLU())
    self.v_linears = torch.nn.Linear(self.key_dim, self.out_dim)

    self.dropout = torch.nn.Dropout(dropout)

    self.merger = MergeLayer(n_node_features, n_node_features, n_node_features, output_dimension)


  def forward(self, node_feature, edge_index, edge_feature, src_time_features, edge_time):
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
    if self.reverse_flag:
      edge_index, edge_feature, src_time_features, edge_time = self.reverse_edge(edge_index,
                                                                                 edge_feature,
                                                                                 src_time_features,
                                                                                 edge_time)
    if self.selfloop_flag:
      edge_index, edge_feature, src_time_features, edge_time = self.add_selfloop(node_feature,
                                                                                 edge_index,
                                                                                 edge_feature,
                                                                                 src_time_features,
                                                                                 edge_time)
    node_i = edge_index[:, 0]
    node_j = edge_index[:, 1]
    node_feat_i = node_feature[node_i, :]
    node_feat_j = node_feature[node_j, :]

    source_node_vec = torch.cat([node_feat_i, src_time_features], dim=1)
    target_node_vec = torch.cat([node_feat_j, edge_feature, edge_time], dim=1)

    q_mat = torch.reshape(self.q_linears(source_node_vec), [-1, self.num_heads, self.d_k])  # [T, N , D]
    k_mat = torch.reshape(self.k_linears(target_node_vec) , [-1, self.num_heads, self.d_k])  # [T, N , D]
    v_mat = torch.reshape(self.v_linears(target_node_vec) , [-1, self.num_heads, self.d_k])  # [T, N , D]

    res_att_sub = torch.sum(torch.multiply(q_mat, k_mat), dim=-1 )* self.scale   #[T, N]

    '''
        Softmax based on target node's id (edge_index_i). Store attention value in self.att.
    '''

    scores = self.scatter_softmax(res_att_sub, node_i)

    # if self.dropout is not None:
    #   scores = self.dropout(scores)

    v = torch.multiply(torch.unsqueeze(scores, dim=2), v_mat)
    v = torch.reshape(v, [-1, self.out_dim])

    out_emb = scatter.scatter_add(v, node_i, dim=0)
    out_emb = self.agg_out(node_feature, out_emb)
    return out_emb


  def scatter_softmax(self, res_att, node_i):
    n_head = self.num_heads
    scores = torch.zeros_like(res_att)
    for i in range(n_head):
      scores[:, i] = scatter.composite.scatter_softmax(res_att[:, i], node_i)

    return scores

  def reverse_edge(self, edge_index, edge_feature, src_time_features, edge_time):
    reverse_edge_index = torch.cat((edge_index[:, 1].unsqueeze(1), edge_index[:, 0].unsqueeze(1)), dim=1)
    two_edge_index = torch.cat((edge_index, reverse_edge_index), dim=0)
    src_time_features = src_time_features.repeat(2, 1)
    edge_feature = edge_feature.repeat(2, 1)
    edge_time = edge_time.repeat(2, 1)

    return two_edge_index, edge_feature, src_time_features, edge_time

  def add_selfloop(self, node_feature, edge_index, edge_feature, src_time_features, edge_time):
    time_emb_unit = src_time_features[0, :].reshape(1, -1)
    node_id = torch.arange(0, node_feature.shape[0], device=edge_index.device).reshape(-1,1)
    edge_index = torch.cat([edge_index, node_id.repeat(1,2)], dim=0)
    edge_feature = torch.cat([edge_feature, torch.zeros([node_id.shape[0], edge_feature.shape[1]], dtype=edge_feature.dtype, device=edge_feature.device)], dim=0)
    src_time_features = torch.cat([src_time_features, time_emb_unit.repeat(node_id.shape[0], 1)], dim=0)
    edge_time = torch.cat([edge_time, time_emb_unit.repeat(node_id.shape[0], 1)], dim=0)

    return edge_index, edge_feature, src_time_features, edge_time


  def agg_out(self, node_feat_pre, node_rep):
    out_embedding = self.merger(node_rep, node_feat_pre)

    return out_embedding