import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        # 第一个图卷积层
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # 第二个图卷积层
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 通过第一个卷积层和 ReLU 激活函数
        # print(x.shape)
        # print(edge_index.shape)
        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # # 通过第二个卷积层
        # x = self.conv2(x, edge_index)
        # return x
        x = x.float()
        batch_size = x.size(0)

        # Iterate over each graph in the batch
        out = []
        for i in range(batch_size):
            x_i = x[i]  # (num_nodes, num_features)
            edge_index_i = edge_index  # Same for all in this example

            # Apply GCN layers
            x_i = self.conv1(x_i, edge_index_i)
            x_i = self.conv2(x_i, edge_index_i)

            out.append(x_i)

        return torch.stack(out)
