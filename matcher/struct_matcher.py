import torch
import networkx as nx
from typing import Dict, Any
from .scoring import ScoringFunction

class StructMatcher:
    """结构匹配器"""
    
    def __init__(self, threshold: float = 0.8, metric: str = "cosine"):
        self.threshold = threshold
        self.metric = metric
        self.scoring = ScoringFunction()
    
    def is_match(self, subgraph_embed: torch.Tensor, 
                 rule_embed: torch.Tensor) -> bool:
        """判断子图是否匹配规则结构"""
        similarity = self.scoring.similarity_score(
            subgraph_embed.unsqueeze(0), 
            rule_embed.unsqueeze(0), 
            self.metric
        )
        return similarity.item() >= self.threshold
    
    def match_score(self, subgraph_embed: torch.Tensor, 
                   rule_embed: torch.Tensor) -> float:
        """计算匹配得分"""
        similarity = self.scoring.similarity_score(
            subgraph_embed.unsqueeze(0), 
            rule_embed.unsqueeze(0), 
            self.metric
        )
        return similarity.item()
    
    def structural_containment_check(self, subgraph: nx.Graph, 
                                   rule_graph: nx.Graph) -> bool:
        """结构包含度检查"""
        # 简单的结构包含检查
        if subgraph.number_of_nodes() < rule_graph.number_of_nodes():
            return False
        if subgraph.number_of_edges() < rule_graph.number_of_edges():
            return False
        
        # 检查度分布
        sub_degrees = sorted([d for n, d in subgraph.degree()])
        rule_degrees = sorted([d for n, d in rule_graph.degree()])
        
        # 简单的度序列包含检查
        for rd in rule_degrees:
            if rd not in sub_degrees:
                return False
        
        return True