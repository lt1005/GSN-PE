# model/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=F.relu):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation

    def forward(self, x, edge_index):
        """
        x: [N, in_dim] 节点特征
        edge_index: [2, E] 边索引 (source, target)
        """
        # 计算每个节点的邻居特征平均（简化版GCN）
        row, col = edge_index  # row: source, col: target
        deg = scatter_add(torch.ones_like(row, dtype=torch.float), col, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 聚合邻居特征
        agg = scatter_add(norm.unsqueeze(1) * x[row], col, dim=0, dim_size=x.size(0))

        out = self.linear(agg)
        if self.activation is not None:
            out = self.activation(out)
        return out


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        N = x.size(0)
        h = self.W(x)  # [N, out_dim]

        row, col = edge_index
        edge_h = torch.cat([h[row], h[col]], dim=1)  # [E, 2*out_dim]

        e = self.leakyrelu(self.a(edge_h)).squeeze()  # [E]

        # 计算attention权重
        attention = torch.zeros(N, N, device=x.device)
        attention[row, col] = e
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        out = torch.matmul(attention, h)
        return out
