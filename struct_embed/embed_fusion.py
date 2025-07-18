import torch
import torch.nn as nn
from typing import Optional

class EmbedFusion(nn.Module):
    """多模态嵌入融合模块，支持motif+GNN+属性三模态"""
    
    def __init__(self, motif_dim: int = 8, gnn_dim: int = 32, attr_dim: int = 0,
                 fusion_dim: int = 40, fusion_type: str = "concat"):
        super().__init__()
        self.fusion_type = fusion_type
        self.attr_dim = attr_dim
        self.motif_dim = motif_dim
        self.gnn_dim = gnn_dim
        input_dim = motif_dim + gnn_dim + attr_dim if attr_dim > 0 else motif_dim + gnn_dim
        if fusion_type == "concat":
            self.output_dim = input_dim
            self.projection = nn.Linear(self.output_dim, fusion_dim)
        elif fusion_type == "weighted":
            self.output_dim = fusion_dim
            self.motif_proj = nn.Linear(motif_dim, fusion_dim)
            self.gnn_proj = nn.Linear(gnn_dim, fusion_dim)
            if attr_dim > 0:
                self.attr_proj = nn.Linear(attr_dim, fusion_dim)
                self.weight_net = nn.Sequential(
                    nn.Linear(motif_dim + gnn_dim + attr_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 3),
                    nn.Softmax(dim=-1)
                )
            else:
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
    
    def forward(self, motif_embed: torch.Tensor, gnn_embed: torch.Tensor, attr_embed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """融合motif、GNN和属性嵌入"""
        if self.fusion_type == "concat":
            if attr_embed is not None:
                fused = torch.cat([motif_embed, gnn_embed, attr_embed], dim=-1)
            else:
                fused = torch.cat([motif_embed, gnn_embed], dim=-1)
            return self.projection(fused)
        elif self.fusion_type == "weighted":
            motif_proj = self.motif_proj(motif_embed)
            gnn_proj = self.gnn_proj(gnn_embed)
            if attr_embed is not None and self.attr_dim > 0:
                attr_proj = self.attr_proj(attr_embed)
                concat_feat = torch.cat([motif_embed, gnn_embed, attr_embed], dim=-1)
                weights = self.weight_net(concat_feat)
                fused = (weights[:, 0:1] * motif_proj +
                         weights[:, 1:2] * gnn_proj +
                         weights[:, 2:3] * attr_proj)
            else:
                concat_feat = torch.cat([motif_embed, gnn_embed], dim=-1)
                weights = self.weight_net(concat_feat)
                fused = weights[:, 0:1] * motif_proj + weights[:, 1:2] * gnn_proj
            return fused
        elif self.fusion_type == "add":
            fused = motif_embed + gnn_embed
            return self.projection(fused)
        else:
            # fallback: 直接拼接
            fused = torch.cat([motif_embed, gnn_embed], dim=-1)
            return self.projection(fused)
