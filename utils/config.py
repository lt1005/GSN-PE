from dataclasses import dataclass
import torch

@dataclass
class Config:
    # 结构嵌入参数
    motif_dim: int = 8
    gnn_hidden_dim: int = 64
    gnn_output_dim: int = 32
    fusion_dim: int = 40
    
    # 语义验证参数
    semantic_dim: int = 128
    semantic_threshold: float = 0.8
    
    # 训练参数
    learning_rate: float = 0.001
    num_epochs: int = 100
    batch_size: int = 32
    
    # 扩展参数
    max_expand_steps: int = 5
    beam_size: int = 3
    similarity_threshold: float = 0.8
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
