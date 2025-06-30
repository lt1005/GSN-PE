import torch
import torch.nn as nn

class EmbedFusion(nn.Module):
    """双模嵌入融合模块"""
    
    def __init__(self, motif_dim: int = 8, gnn_dim: int = 32, 
                 fusion_dim: int = 40, fusion_type: str = "concat"):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            self.output_dim = motif_dim + gnn_dim
            self.projection = nn.Linear(self.output_dim, fusion_dim)
        elif fusion_type == "weighted":
            self.output_dim = fusion_dim
            self.motif_proj = nn.Linear(motif_dim, fusion_dim)
            self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
            self.weight_net = nn.Sequential(
                nn.Linear(motif_dim + gnn_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
                nn.Softmax(dim=-1)
            )
        elif fusion_type == "add":
            assert motif_dim == gnn_dim, "Add fusion requires same dimensions"
            self.output_dim = motif_dim
            self.projection = nn.Linear(motif_dim, fusion_dim)
    
    def forward(self, motif_embed: torch.Tensor, gnn_embed: torch.Tensor) -> torch.Tensor:
        """融合motif和GNN嵌入"""
        if self.fusion_type == "concat":
            fused = torch.cat([motif_embed, gnn_embed], dim=-1)
            return self.projection(fused)
        
        elif self.fusion_type == "weighted":
            motif_proj = self.motif_proj(motif_embed)
            gnn_proj = self.gnn_proj(gnn_embed)
            
            # 计算权重
            concat_feat = torch.cat([motif_embed, gnn_embed], dim=-1)
            weights = self.weight_net(concat_feat)
            
            # 加权融合
            fused = weights[:, 0:1] * motif_proj + weights[:, 1:2] * gnn_proj
            return fused
        
        elif self.fusion_type == "add":
            fused = motif_embed + gnn_embed
            return self.projection(fused)
