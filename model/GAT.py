import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super(GAT, self).__init__()
        # 第一个 GAT 层
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        # 第二个 GAT 层
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        # 通过第一个 GAT 层和 ReLU 激活函数
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        # 通过第二个 GAT 层
        x = self.gat2(x, edge_index)
        return x
