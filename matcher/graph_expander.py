import networkx as nx
import torch
from typing import List, Tuple, Set, Optional
import heapq
from collections import defaultdict
import logging

class GraphExpander:
    """子图扩展器"""
    
    def __init__(self, max_steps: int = 5, beam_size: int = 3):
        self.max_steps = max_steps
        self.beam_size = beam_size
    
    def expand_from_anchor(self, anchor: Tuple[str, str, str], 
                          kg: nx.MultiDiGraph, 
                          target_embed: torch.Tensor,
                          embed_model) -> nx.Graph:
        """从anchor三元组开始扩展子图"""
        h, r, t = anchor
        
        # 初始化子图
        current_subgraph = nx.Graph()
        current_subgraph.add_edge(h, t, relation=r)
        
        # 候选扩展列表 (priority_queue)
        candidates = []
        visited_edges = {(h, t, r)}
        
        for step in range(self.max_steps):
            # 获取当前子图的嵌入
            current_embed = self._get_graph_embedding(current_subgraph, embed_model)
            current_score = self._similarity_score(current_embed, target_embed)
            
            # 找到所有可能的扩展边
            expansion_edges = self._find_expansion_edges(
                current_subgraph, kg, visited_edges
            )
            
            if not expansion_edges:
                break
            
            # 评估每个扩展选项
            best_expansions = []
            for edge in expansion_edges[:self.beam_size * 3]:  # 限制候选数量
                temp_subgraph = current_subgraph.copy()
                eh, et, er = edge
                temp_subgraph.add_edge(eh, et, relation=er)
                
                temp_embed = self._get_graph_embedding(temp_subgraph, embed_model)
                temp_score = self._similarity_score(temp_embed, target_embed)
                
                heapq.heappush(best_expansions, (-temp_score, edge, temp_subgraph))
            
            # 选择最佳扩展
            if best_expansions:
                _, best_edge, best_subgraph = heapq.heappop(best_expansions)
                current_subgraph = best_subgraph
                visited_edges.add(best_edge)
                
                logging.info(f"步骤 {step+1}: 添加边 {best_edge}, 当前相似度: {current_score:.4f}")
            else:
                break
        
        return current_subgraph
    
    def _find_expansion_edges(self, subgraph: nx.Graph, 
                            kg: nx.MultiDiGraph, 
                            visited: Set) -> List[Tuple[str, str, str]]:
        """找到可能的扩展边"""
        candidates = []
        subgraph_nodes = set(subgraph.nodes())
        
        # 从子图节点出发寻找新边
        for node in subgraph_nodes:
            # 出边
            if node in kg:
                for neighbor in kg.neighbors(node):
                    for edge_data in kg[node][neighbor].values():
                        relation = edge_data.get('relation', 'unknown')
                        edge = (node, neighbor, relation)
                        if edge not in visited:
                            candidates.append(edge)
            
            # 入边
            for pred in kg.predecessors(node):
                for edge_data in kg[pred][node].values():
                    relation = edge_data.get('relation', 'unknown')
                    edge = (pred, node, relation)
                    if edge not in visited:
                        candidates.append(edge)
        
        return candidates
    
    def _get_graph_embedding(self, graph: nx.Graph, embed_model) -> torch.Tensor:
        """获取图的嵌入表示"""
        # 这里需要调用嵌入模型
        # 简化实现，实际需要与motif提取器和GNN结合
        from ..utils.graph_utils import nx_to_pyg
        from ..struct_embed.motif_extractor import MotifExtractor
        
        motif_extractor = MotifExtractor()
        motif_embed = motif_extractor.extract_motifs(graph)
        
        # 如果图为空，返回零向量
        if graph.number_of_nodes() == 0:
            return torch.zeros_like(motif_embed)
        
        return motif_embed
    
    def _similarity_score(self, embed1: torch.Tensor, embed2: torch.Tensor) -> float:
        """计算嵌入相似度"""
        return torch.cosine_similarity(embed1.unsqueeze(0), embed2.unsqueeze(0)).item()