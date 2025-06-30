import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from typing import Tuple, Optional, Dict, Any
import logging

class SemanticVerifier:
    """语义验证模块 - 基于SRS实现"""
    
    def __init__(self, entity_encoder=None, relation_encoder=None, 
                 threshold: float = 0.8, semantic_dim: int = 128):
        self.entity_encoder = entity_encoder
        self.relation_encoder = relation_encoder
        self.threshold = threshold
        self.semantic_dim = semantic_dim
        self.logger = logging.getLogger(__name__)
        
        # 如果没有提供编码器，使用简单的随机初始化
        if self.entity_encoder is None:
            self.entity_encoder = self._create_simple_entity_encoder()
        if self.relation_encoder is None:
            self.relation_encoder = self._create_simple_relation_encoder()
    
    def _create_simple_entity_encoder(self) -> nn.Module:
        """创建简单的实体编码器"""
        return nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, self.semantic_dim)
        )
    
    def _create_simple_relation_encoder(self) -> nn.Module:
        """创建简单的关系编码器"""
        return nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, self.semantic_dim)
        )
    
    def encode_semantic(self, subgraph: nx.Graph) -> torch.Tensor:
        """
        输入：
            subgraph: 子图对象（包含实体节点、关系边）
        输出：
            语义嵌入向量（Tensor）
        """
        try:
            if subgraph.number_of_nodes() == 0:
                return torch.zeros(self.semantic_dim)
            
            # 提取实体和关系
            entities = list(subgraph.nodes())
            edges = list(subgraph.edges(data=True))
            
            # 编码实体
            entity_embeddings = []
            for entity in entities:
                # 简单的实体编码：使用实体名的哈希值作为特征
                entity_feature = torch.tensor([hash(entity) % 1000 / 1000.0], dtype=torch.float32)
                entity_embed = self.entity_encoder(entity_feature)
                entity_embeddings.append(entity_embed)
            
            # 编码关系
            relation_embeddings = []
            for _, _, edge_data in edges:
                relation = edge_data.get('relation', 'unknown')
                # 简单的关系编码：使用关系名的哈希值作为特征
                relation_feature = torch.tensor([hash(relation) % 1000 / 1000.0], dtype=torch.float32)
                relation_embed = self.relation_encoder(relation_feature)
                relation_embeddings.append(relation_embed)
            
            # 聚合所有嵌入
            all_embeddings = entity_embeddings + relation_embeddings
            if all_embeddings:
                # 使用平均池化
                semantic_embed = torch.stack(all_embeddings).mean(dim=0)
            else:
                semantic_embed = torch.zeros(self.semantic_dim)
            
            return semantic_embed
            
        except Exception as e:
            self.logger.error(f"语义编码失败: {e}")
            # 返回零向量，避免语义验证误接受
            return torch.zeros(self.semantic_dim)
    
    def semantic_similarity(self, semantic_vec1: torch.Tensor, 
                           semantic_vec2: torch.Tensor) -> float:
        """
        计算两个语义嵌入向量间相似度得分（余弦相似度）
        """
        try:
            # 确保向量维度一致
            if semantic_vec1.shape != semantic_vec2.shape:
                min_dim = min(semantic_vec1.shape[0], semantic_vec2.shape[0])
                semantic_vec1 = semantic_vec1[:min_dim]
                semantic_vec2 = semantic_vec2[:min_dim]
            
            # 计算余弦相似度
            similarity = F.cosine_similarity(semantic_vec1.unsqueeze(0), 
                                           semantic_vec2.unsqueeze(0), dim=1)
            return similarity.item()
            
        except Exception as e:
            self.logger.error(f"语义相似度计算失败: {e}")
            return 0.0
    
    def semantic_verify(self, subgraph: nx.Graph, rule_semantic_vec: torch.Tensor) -> Tuple[bool, float]:
        """
        对子图进行语义验证：
        输入：
            subgraph - 待验证子图
            rule_semantic_vec - 规则结构的语义嵌入
        输出：
            是否语义匹配通过（bool）
            语义相似度得分（float）
        """
        try:
            # 编码子图语义
            subgraph_semantic_vec = self.encode_semantic(subgraph)
            
            # 计算语义相似度
            semantic_score = self.semantic_similarity(subgraph_semantic_vec, rule_semantic_vec)
            
            # 判断是否通过验证
            is_passed = semantic_score >= self.threshold
            
            return is_passed, semantic_score
            
        except Exception as e:
            self.logger.error(f"语义验证失败: {e}")
            # 验证失败时返回False，避免误接受
            return False, 0.0
    
    def batch_semantic_verify(self, subgraphs: list, rule_semantic_vec: torch.Tensor) -> list:
        """
        批量语义验证
        输入：
            subgraphs - 子图列表
            rule_semantic_vec - 规则语义嵌入
        输出：
            验证结果列表 [(is_passed, score), ...]
        """
        results = []
        for subgraph in subgraphs:
            result = self.semantic_verify(subgraph, rule_semantic_vec)
            results.append(result)
        return results 