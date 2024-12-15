import sys

import torch
import torch.utils.data
import os
import numpy as np
from option import args
import random
import pandas as pd
from utils import get_neighbor_finder
from operator import itemgetter
import pickle


class DygDataset(torch.utils.data.Dataset):
    def __init__(self, config, split_flag):
        self.config = config

        dataset_name = '{}/{}.pkl'.format(config.dir_data, config.data_set)

        with open(dataset_name, 'rb') as file:
            data = pickle.load(file)
        self.input_nodes_feature, self.input_edges_feature, self.input_edges_pad, self.labels, self.Tmats, self.adjs, self.eadjs, self.mask_edge = self.get_data(data, split_flag)
    def get_data(self, data, split_flag):
        node_features = data['nodefeatures']
        edge_features = data['edgefeatures']
        labels = data['labels']
        Tmats = data['Tmats']
        adjs = data['adjs']
        eadjs = data['eadjs']

        flattened_node = np.concatenate([arr for arr in node_features])
        unique_node = np.unique(flattened_node)
        num_nodes = int(max(unique_node.size, np.max(unique_node) + 1))

        flattened_edge = np.concatenate([arr for arr in edge_features])
        unique_edge = np.unique(flattened_edge)
        num_edges = unique_edge.size

        Nfeatures = np.random.uniform(low=0.0, high=1.0, size=(num_nodes, self.config.input_dim))
        Efeatures = np.random.uniform(low=0.0, high=1.0, size=(num_edges, self.config.input_dim))

        max_mask_edge = max(len(arr) for arr in edge_features)
        masks_edge = [len(edge) for edge in edge_features]

        NS = len(edge_features)
        mask_edge = np.ones((NS, max_mask_edge))
        for i in range(NS):
            mask_edge[i, :masks_edge[i]] = 0

        hidden = Efeatures.shape[1]
        input_edges_pad = np.zeros((NS, max_mask_edge, hidden))
        for i, indices in enumerate(edge_features):
            indices = indices.astype(int)
            input_edges_pad[i, :len(indices), :] = Efeatures[indices]

        input_edges_feature = [torch.tensor(input_edges_pad[i, :masks_edge[i], :]) for i in range(NS)]

        max_mask_node = max(len(arr) for arr in node_features)
        masks_node = [len(node) for node in node_features]


        NS = len(node_features)
        hidden = Nfeatures.shape[1]
        input_nodes_pad = np.zeros((NS, max_mask_node, hidden))
        for i, indices in enumerate(node_features):
            indices = indices.astype(int)
            input_nodes_pad[i, :len(indices), :] = Nfeatures[indices]

        input_nodes_feature = [torch.tensor(input_nodes_pad[i, :masks_node[i], :]) for i in range(NS)]
        if self.config.data_set == 'btc_alpha':
            split_indices = 7000
        elif self.config.data_set == 'btc_otc':
            split_indices = 10000

        if split_flag == 'train':
            input_nodes_feature = input_nodes_feature[:split_indices]
            input_edges_feature = input_edges_feature[:split_indices]
            input_edges_pad = input_edges_pad[:split_indices]
            labels = labels[:split_indices]
            Tmats = Tmats[:split_indices]
            adjs = adjs[:split_indices]
            eadjs = eadjs[:split_indices]
            mask_edge = mask_edge[:split_indices]
        elif split_flag == 'test':
            input_nodes_feature = input_nodes_feature[split_indices:]
            input_edges_feature = input_edges_feature[split_indices:]
            input_edges_pad = input_edges_pad[split_indices:]
            labels = labels[split_indices:]
            Tmats = Tmats[split_indices:]
            adjs = adjs[split_indices:]
            eadjs = eadjs[split_indices:]
            mask_edge = mask_edge[split_indices:]

        input_edges_pad = torch.tensor(input_edges_pad)
        labels = torch.tensor(labels)
        mask_edge = torch.tensor(mask_edge)
        return input_nodes_feature, input_edges_feature, input_edges_pad, labels, Tmats, adjs, eadjs, mask_edge

    def __getitem__(self, item):

        sinput_nodes_feature = self.input_nodes_feature[item]
        sinput_edges_feature = self.input_edges_feature[item]
        sinput_edges_pad = self.input_edges_pad[item]
        slabels = self.labels[item]
        sTmats = self.Tmats[item]
        sadjs = self.adjs[item]
        seadjs = self.eadjs[item]
        smask_edge = self.mask_edge[item]

        return {
            'input_nodes_feature': sinput_nodes_feature,
            'input_edges_feature': sinput_edges_feature,
            'input_edges_pad': sinput_edges_pad,
            'labels': slabels,
            'Tmats': sTmats,
            'adjs': sadjs,
            'eadjs': seadjs,
            'mask_edge':smask_edge
        }

    def __len__(self):
        return len(self.labels)


class Collate:
    def __init__(self, config):
        self.config = config

    def dyg_collate_fn(self, batch):
        input_nodes_feature = [b['input_nodes_feature'] for b in batch]
        input_edges_feature = [b['input_edges_feature'] for b in batch]
        input_edges_pad = torch.stack([b['input_edges_pad'] for b in batch], dim=0)

        labels = torch.stack([b['labels'] for b in batch], dim=0)

        Tmats = [b['Tmats'] for b in batch]
        adjs = [b['adjs'] for b in batch]
        eadjs = [b['eadjs'] for b in batch]
        mask_edge = torch.stack([b['mask_edge'] for b in batch], dim=0)

        return {
            'input_nodes_feature': input_nodes_feature,
            'input_edges_feature': input_edges_feature,
            'input_edges_pad': input_edges_pad,
            'labels': labels,
            'Tmats': Tmats,
            'adjs': adjs,
            'eadjs': eadjs,
            'mask_edge':mask_edge
        }


if __name__ == '__main__':
    config = args
    a = DygDataset(config, 'train')
    # a = DygDatasetTest(config, 'val')
    c = a[5000]
    print(c)
