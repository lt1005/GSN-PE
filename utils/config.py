from dataclasses import dataclass
import torch

@dataclass
class Config:
    # 结构嵌入参数
    motif_types: list = None  # type: ignore  # 将在__post_init__中设置默认值
    motif_dim: int = 8  # 原始motif维度
    gnn_hidden_dim: int = 64
    gnn_output_dim: int = 64  # 与GNN编码器实际输出维度匹配
    fusion_dim: int = 40
    
    # 语义验证参数
    embedding_dim: int = 100  # 实体和关系的embedding维度
    semantic_feature_dim: int = 64  # 语义特征维度
    semantic_dim: int = 128
    semantic_threshold: float = 0.5  # 降低语义阈值从0.8到0.5
    structure_weight: float = 0.7
    semantic_weight: float = 0.3
    
    # 训练参数
    learning_rate: float = 0.001
    num_epochs: int = 100
    batch_size: int = 32
    
    # 扩展参数
    max_expand_steps: int = 5
    beam_size: int = 3
    similarity_threshold: float = 0.7  # 降低结构相似度阈值从0.8到0.7
    final_threshold: float = 0.5  # 降低最终匹配阈值从0.6到0.5
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.motif_types is None:
            self.motif_types = [
                "triangle", "star", "chain", "cycle", 
                "clique", "path", "tree", "bipartite"
            ]
