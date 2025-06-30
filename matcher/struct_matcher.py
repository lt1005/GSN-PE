import torch
import torch.nn.functional as F
import networkx as nx
from typing import Tuple, List, Dict, Any
import logging

class StructMatcher:
    """结构匹配模块 - 基于SRS实现"""
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
    
    def is_match(self, subgraph_embed: torch.Tensor, rule_embed: torch.Tensor) -> bool:
        """
        判断候选子图与规则结构是否匹配
        输入：候选子图嵌入与目标结构嵌入
        输出：是否匹配
        """
        try:
            similarity = self.match_score(subgraph_embed, rule_embed)
            return similarity >= self.threshold
        except Exception as e:
            self.logger.error(f"结构匹配判断失败: {e}")
            return False
    
    def match_score(self, subgraph_embed: torch.Tensor, rule_embed: torch.Tensor) -> float:
        """
        计算匹配得分（距离或相似度）
        输入：两个结构嵌入向量
        输出：距离或相似度分数
        """
        try:
            # 使用余弦相似度
            similarity = F.cosine_similarity(
                subgraph_embed.unsqueeze(0), 
                rule_embed.unsqueeze(0), 
                dim=1
            )
            return similarity.item()
        except Exception as e:
            self.logger.error(f"匹配得分计算失败: {e}")
            return 0.0
    
    def batch_match(self, subgraph_embeds: List[torch.Tensor], 
                   rule_embed: torch.Tensor) -> List[Tuple[bool, float]]:
        """
        批量结构匹配
        输入：候选子图嵌入列表与目标结构嵌入
        输出：匹配结果列表 [(is_match, score), ...]
        """
        results = []
        for subgraph_embed in subgraph_embeds:
            is_match = self.is_match(subgraph_embed, rule_embed)
            score = self.match_score(subgraph_embed, rule_embed)
            results.append((is_match, score))
        return results
    
    def isomorphic_match(self, subgraph: nx.Graph, rule_graph: nx.Graph) -> bool:
        """
        精确同构检测
        输入：候选子图与规则图
        输出：是否同构
        """
        try:
            # 使用NetworkX的图同构检测
            return nx.is_isomorphic(subgraph, rule_graph)
        except Exception as e:
            self.logger.error(f"同构检测失败: {e}")
            return False
    
    def structural_similarity(self, subgraph: nx.Graph, rule_graph: nx.Graph) -> float:
        """
        计算结构相似度（基于图特征）
        输入：候选子图与规则图
        输出：结构相似度分数
        """
        try:
            # 计算基本图特征
            sub_nodes = subgraph.number_of_nodes()
            sub_edges = subgraph.number_of_edges()
            rule_nodes = rule_graph.number_of_nodes()
            rule_edges = rule_graph.number_of_edges()
            
            # 节点数和边数的相似度
            node_sim = 1.0 - abs(sub_nodes - rule_nodes) / max(sub_nodes, rule_nodes, 1)
            edge_sim = 1.0 - abs(sub_edges - rule_edges) / max(sub_edges, rule_edges, 1)
            
            # 平均相似度
            return (node_sim + edge_sim) / 2.0
            
        except Exception as e:
            self.logger.error(f"结构相似度计算失败: {e}")
            return 0.0