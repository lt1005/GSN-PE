# model/gnn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GCNLayer

class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super(GNNEncoder, self).__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        """
        x: [N, in_dim] 节点特征
        edge_index: [2, E] 边索引
        batch: [N]，批次中每个节点所属图的索引（可选）
        """
        x = self.gcn1(x, edge_index)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)

        # 池化节点向量得到图嵌入，简单用mean pooling
        if batch is not None:
            # batch 是节点对应的图索引，按图聚合
            import torch_scatter
            graph_emb = torch_scatter.scatter_mean(x, batch, dim=0)
        else:
            graph_emb = x.mean(dim=0, keepdim=True)  # 单个图时取平均

        return graph_emb  # [num_graphs, out_dim] 或 [1, out_dim]
