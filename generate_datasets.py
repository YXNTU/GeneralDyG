from tqdm import tqdm
import pickle
from option import args
import pandas as pd
import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
import sys
import os
import random
import copy
import matplotlib.pyplot as plt

# dataset_name = '{}/ml_{}'.format(config.dir_data, config.data_set)
# graph_df = pd.read_csv('{}.csv'.format(dataset_name))
# edge_features = np.load('{}.npy'.format(dataset_name))
# node_features = np.load('{}_node.npy'.format(dataset_name))

class BatchGraphSample:
    def __init__(self, config, graph_df):
        self.config = config
        self.graph_df = graph_df
        # 生成图
        self.G = nx.Graph()
        for index, row in graph_df.iterrows():
            u, v, weight = row['u'], row['i'], row['id']
            if self.G.has_edge(u, v):
                # 如果边已经存在，将新权重添加到现有权重列表
                self.G[u][v]['weight'].append(weight)
            else:
                # 如果边不存在，创建新的边并初始化权重列表
                self.G.add_edge(u, v, weight=[weight])
        # # k = 1
        # if self.config.data_set == 'btc_otc':
        #     self.max_mask_len = 14
        # elif self.config.data_set == 'btc_alpha':
        #     self.max_mask_len = 16

        # k = 1
        if self.config.data_set == 'btc_otc':
            self.max_mask_len = 26
        elif self.config.data_set == 'btc_alpha':
            self.max_mask_len = 26

        # k = 2
        # if self.config.data_set == 'btc_otc':
        #     self.max_mask_len = 20
        # elif self.config.data_set == 'btc_alpha':
        #     self.max_mask_len = 20


    def remove_random_nodes(self, nodes, max_mask_len, src_node, dest_node):
        nodes = list(nodes)
        # 确保node在nodes中
        if src_node not in nodes:
            raise ValueError("The required 'node' is not in the list 'nodes'")

            # 当nodes的长度超过max_mask_len时，开始随机删除
        while len(nodes) > max_mask_len:
            # 随机选择一个要删除的元素，但不能是node
            to_remove = random.choice(nodes)
            if to_remove != src_node and to_remove != dest_node:
                nodes.remove(to_remove)
        return set(nodes)

        # 提取k-hop子图的函数

    def extract_k_hop_subgraph(self, graph, src_node, dest_node, k, max_mask_len):
        nodes = set([src_node])
        visited = set([src_node])
        for _ in range(k):
            new_nodes = set()
            for n in nodes:
                neighbors = set(graph.neighbors(n))
                new_nodes.update(neighbors - visited)
            visited.update(new_nodes)
            nodes.update(new_nodes)
        nodes = self.remove_random_nodes(nodes, max_mask_len, src_node, dest_node)
        return nodes

        # 根据权重和时间戳重新排序节点

    def reorder_nodes(self, subgraph):
        node_weights = {}
        for node in subgraph.nodes:
            edges = subgraph.edges(node, data='weight')
            if edges:
                node_weights[node] = min(weight for _, _, weight in edges)
            else:
                node_weights[node] = float('inf')
        sorted_nodes = sorted(node_weights.items(), key=lambda x: x[1])
        node_mapping = {old_id: new_id for new_id, (old_id, _) in enumerate(sorted_nodes)}
        return node_mapping

        # 替换子图中的节点和边

    def replace_subgraph(self, subgraph, node_mapping):
        new_subgraph = nx.Graph()
        new_edge_features = {}

        num_nodes = len(subgraph.nodes())
        new_node_features = np.zeros(num_nodes)

        for node, new_node_id in node_mapping.items():
            new_node_features[new_node_id] = int(node)
            new_subgraph.add_node(new_node_id)

        mask = new_node_features.shape[0]

        for edge in subgraph.edges(data=True):
            new_source = node_mapping[edge[0]]
            new_target = node_mapping[edge[1]]
            new_subgraph.add_edge(new_source, new_target, weight=1)
            new_edge_features[(new_source, new_target)] = int(edge[2]['weight'])

        return new_subgraph, new_node_features, new_edge_features, mask

    def create_transition_matrix(self, vertex_adj):
        '''create N_v * N_e transition matrix'''
        vertex_adj.setdiag(0)
        edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
        num_edge = int(len(edge_index[0]))
        edge_name = [x for x in zip(edge_index[0], edge_index[1])]

        row_index = [i for sub in edge_name for i in sub]
        col_index = np.repeat([i for i in range(num_edge)], 2)

        data = np.ones(num_edge * 2)
        T = sp.csr_matrix((data, (row_index, col_index)),
                          shape=(vertex_adj.shape[0], num_edge))

        return T

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def create_edge_adj(self, vertex_adj):
        '''
        create an edge adjacency matrix from vertex adjacency matrix
        '''
        vertex_adj.setdiag(0)
        edge_index = np.nonzero(sp.triu(vertex_adj, k=1))
        num_edge = int(len(edge_index[0]))
        edge_name = [x for x in zip(edge_index[0], edge_index[1])]

        edge_adj = np.zeros((num_edge, num_edge))
        for i in range(num_edge):
            for j in range(i, num_edge):
                if len(set(edge_name[i]) & set(edge_name[j])) == 0:
                    edge_adj[i, j] = 0
                else:
                    edge_adj[i, j] = 1
        adj = edge_adj + edge_adj.T
        np.fill_diagonal(adj, 1)
        return sp.csr_matrix(adj), edge_name

        # 假设 new_subgraph 是一个 networkx.Graph 对象

    def get_adjacency_matrix(self, subgraph):
        # 获取邻接矩阵，返回的是 SciPy 稀疏矩阵
        adj_matrix_sparse = nx.adjacency_matrix(subgraph)
        # 将稀疏矩阵转换为 NumPy 数组
        adj_matrix = adj_matrix_sparse.toarray()
        return adj_matrix

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1)).astype("float")
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def pad_ndarray_list(self, ndarray_list, max_mask_len):
        # 获取列表的大小
        N = len(ndarray_list)
        hidden = ndarray_list[0].shape[1]  # 假设hidden维度是固定的

        # 初始化填充后的三维ndarray和掩码矩阵
        padded_ndarray = np.zeros((N, max_mask_len, hidden))
        mask = np.ones((N, max_mask_len), dtype=bool)  # 填充部分为True，原始部分为False
        unpadded_indices = []

        for i, array in enumerate(ndarray_list):
            seq_len = array.shape[0]
            padded_ndarray[i, :seq_len, :] = array
            mask[i, :seq_len] = False  # 原始部分为False
            unpadded_indices.append(seq_len)  # 保存每个ndarray的原始长度

        unpadded_indices = torch.tensor(unpadded_indices)  # 转换为tensor

        return padded_ndarray, mask, unpadded_indices


    def get_batch_data(self, idx_batch):
        k = 1
        all_node_features = []
        all_edge_features = []
        all_Tmat = []
        all_adj = []
        all_eadj = []
        all_mask = []

        edges = self.graph_df[self.graph_df['id'].isin(idx_batch)]
        # for index, edge in edges.iterrows():
        for index, edge in tqdm(edges.iterrows(), total=len(edges)):
            src_nodes = self.extract_k_hop_subgraph(copy.deepcopy(self.G), edge['u'], edge['i'], k, self.max_mask_len)
            dest_nodes = self.extract_k_hop_subgraph(copy.deepcopy(self.G), edge['i'], edge['u'], k, self.max_mask_len)
            subgraph_nodes = src_nodes.union(dest_nodes)
            batch_subgraph = copy.deepcopy(self.G).subgraph(subgraph_nodes)
            for u, v, data in batch_subgraph.edges(data=True):
                # 从权重列表中随机选择一个数作为权重
                random_weight = random.choice(data['weight'])
                # 更新边的权重为随机选择的值
                batch_subgraph[u][v]['weight'] = random_weight
            batch_subgraph[edge['u']][edge['i']]['weight'] = edge['id']

            node_mapping = self.reorder_nodes(batch_subgraph)
            new_subgraph, new_node_features, new_edge_features, mask = self.replace_subgraph(batch_subgraph,
                                                                                             node_mapping)
            adj = nx.adjacency_matrix(new_subgraph)
            T = self.create_transition_matrix(adj)
            tensor_T = self.sparse_mx_to_torch_sparse_tensor(T)
            eadj, edge_name = self.create_edge_adj(adj)

            num_edges = eadj.shape[0]  # 非零元素的数量（即边的数量）
            edge_feature_matrix = np.zeros(num_edges)
            edge_idx = 0
            temp_T = T.copy().toarray()
            for col in range(temp_T.shape[1]):
                indices = np.where(temp_T[:, col] == 1)[0]
                i, j = indices[0], indices[1]
                if (i, j) in new_edge_features:
                    edge_feature_matrix[edge_idx] = new_edge_features[(i, j)]
                else:
                    edge_feature_matrix[edge_idx] = new_edge_features[(j, i)]
                edge_idx += 1
            eadj = self.sparse_mx_to_torch_sparse_tensor(self.normalize(eadj))

            adj = self.sparse_mx_to_torch_sparse_tensor(self.normalize(adj + sp.eye(adj.shape[0])))

            all_node_features.append(new_node_features)
            all_edge_features.append(edge_feature_matrix)
            all_Tmat.append(tensor_T)
            all_adj.append(adj)
            all_eadj.append(eadj)
            all_mask.append(mask)
        return  all_node_features, all_edge_features, all_Tmat, all_adj, all_eadj, all_mask


if __name__ == '__main__':
    # import pickle
    # # def torch_sparse_tensor_to_scipy(torch_sparse_tensor):
    # #     """Convert a torch sparse tensor to a scipy sparse matrix."""
    # #     indices = torch_sparse_tensor._indices().numpy()
    # #     values = torch_sparse_tensor._values().numpy()
    # #     shape = torch_sparse_tensor.shape
    # #     return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
    #
    config = args
    #
    # dataset_name = '{}/{}.pkl'.format(config.dir_data, config.data_set)
    #
    # with open(dataset_name, 'rb') as file:
    #     data = pickle.load(file)
    #
    # node_features = data['nodefeatures']
    # edge_features = data['edgefeatures']
    # labels = data['labels']
    # Tmats = data['Tmats']
    # adjs = data['adjs']
    # eadjs = data['eadjs']
    #
    # # dense_Tmats = [
    # #     torch.tensor(torch_sparse_tensor_to_scipy(sparse_tensor).toarray())
    # #     for sparse_tensor in Tmats
    # # ]
    # # dense_adjs = [
    # #     torch.tensor(torch_sparse_tensor_to_scipy(sparse_tensor).toarray())
    # #     for sparse_tensor in adjs
    # # ]
    # # dense_eadjs = [
    # #     torch.tensor(torch_sparse_tensor_to_scipy(sparse_tensor).toarray())
    # #     for sparse_tensor in eadjs
    # # ]
    #
    #
    # new_data = {
    #     'nodefeatures': node_features,
    #     'edgefeatures': edge_features,
    #     'labels': labels,
    #     'Tmats': Tmats,
    #     'adjs': adjs,
    #     'eadjs': eadjs
    # }
    # with open(config.data_set + '.pkl', 'wb') as f:
    #     pickle.dump(new_data, f)
    # print('装载新数据完成')



    # sys.exit()
    #
    dataset_name = '{}/{}_0.5_0.{}'.format(config.dir_data, config.data_set, config.neg)
    graph_df = pd.read_csv('{}.csv'.format(dataset_name))
    label_column = graph_df['label'].to_numpy()
    idx_column = graph_df['id'].to_numpy()
    a = BatchGraphSample(config, graph_df)
    all_node_features, all_edge_features, all_Tmat, all_adj, all_eadj, all_mask = a.get_batch_data(
        idx_column)
    all_node_features = np.array(all_node_features)
    all_edge_features = np.array(all_edge_features)
    all_mask = np.array(all_mask)

    data = {
        'nodefeatures': all_node_features,
        'edgefeatures': all_edge_features,
        'masks': all_mask,
        'labels': label_column,
        'Tmats': all_Tmat,
        'adjs': all_adj,
        'eadjs': all_eadj
    }
    with open(config.data_set + '.pkl', 'wb') as f:
        pickle.dump(data, f)

    print('-----------------------------------')
