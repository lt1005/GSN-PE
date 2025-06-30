import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GATConv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

class GNNEncoder(nn.Module):
    """GNN结构嵌入模块"""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, output_dim: int = 32, 
                 gnn_type: str = "GIN", num_layers: int = 3):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        
        # 构建GNN层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # 第一层
        if gnn_type == "GIN":
            nn_layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_layer))
        elif gnn_type == "GAT":
            self.convs.append(GATConv(input_dim, hidden_dim))
        elif gnn_type == "GCN":
            self.convs.append(GCNConv(input_dim, hidden_dim))
        
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            if gnn_type == "GIN":
                nn_layer = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(nn_layer))
            elif gnn_type == "GAT":
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            elif gnn_type == "GCN":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 最后一层
        if num_layers > 1:
            if gnn_type == "GIN":
                nn_layer = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
                self.convs.append(GINConv(nn_layer))
            elif gnn_type == "GAT":
                self.convs.append(GATConv(hidden_dim, output_dim))
            elif gnn_type == "GCN":
                self.convs.append(GCNConv(hidden_dim, output_dim))
            
            self.batch_norms.append(nn.BatchNorm1d(output_dim))
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, data: Data) -> torch.Tensor:
        """前向传播"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        print("x shape:", x.shape)
        print("edge_index shape:", edge_index.shape)
        
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        # GNN层
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # 图级池化
        graph_embed = global_mean_pool(x, batch)
        
        return graph_embed