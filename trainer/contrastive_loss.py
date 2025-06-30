import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    
    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        embeddings: (batch_size, embed_dim)
        labels: (batch_size,) 结构类型标签
        """
        batch_size = embeddings.size(0)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        # 创建正负样本掩码
        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.t()).float()
        neg_mask = (labels != labels.t()).float()
        
        # 去除对角线（自己与自己的相似度）
        pos_mask.fill_diagonal_(0)
        
        # 计算对比损失
        pos_sim = sim_matrix * pos_mask
        neg_sim = sim_matrix * neg_mask
        
        # InfoNCE损失
        pos_exp = torch.exp(pos_sim).sum(dim=1)
        all_exp = torch.exp(sim_matrix).sum(dim=1)
        
        loss = -torch.log(pos_exp / (all_exp + 1e-8)).mean()
        
        return loss
    
    def triplet_loss(self, anchor: torch.Tensor, 
                    positive: torch.Tensor, 
                    negative: torch.Tensor) -> torch.Tensor:
        """三元组损失"""
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        return loss
