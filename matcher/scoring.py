import torch
import torch.nn.functional as F

class ScoringFunction:
    """嵌入距离度量与评分函数"""
    
    @staticmethod
    def cosine_similarity(embed1: torch.Tensor, embed2: torch.Tensor) -> torch.Tensor:
        """余弦相似度"""
        return F.cosine_similarity(embed1, embed2, dim=-1)
    
    @staticmethod
    def l2_distance(embed1: torch.Tensor, embed2: torch.Tensor) -> torch.Tensor:
        """L2距离"""
        return torch.norm(embed1 - embed2, p=2, dim=-1)
    
    @staticmethod
    def l1_distance(embed1: torch.Tensor, embed2: torch.Tensor) -> torch.Tensor:
        """L1距离"""
        return torch.norm(embed1 - embed2, p=1, dim=-1)
    
    @staticmethod
    def similarity_score(embed1: torch.Tensor, embed2: torch.Tensor, 
                        metric: str = "cosine") -> torch.Tensor:
        """计算相似度得分"""
        if metric == "cosine":
            return ScoringFunction.cosine_similarity(embed1, embed2)
        elif metric == "l2":
            return -ScoringFunction.l2_distance(embed1, embed2)  # 负距离作为相似度
        elif metric == "l1":
            return -ScoringFunction.l1_distance(embed1, embed2)
        else:
            raise ValueError(f"Unknown metric: {metric}")