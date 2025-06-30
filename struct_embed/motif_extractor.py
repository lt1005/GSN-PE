import networkx as nx
import torch
import numpy as np
from typing import Tuple
from collections import defaultdict

class MotifExtractor:
    """motif-count向量提取器"""
    
    def __init__(self, motif_dim: int = 8):
        self.motif_dim = motif_dim
        self.motif_types = [
            'path_2', 'fork', 'triangle', 'star_in', 
            'star_out', 'cycle_3', 'cycle_4', 'complete_3'
        ]
    
    def extract_motifs(self, G: nx.Graph) -> torch.Tensor:
        """提取图的motif特征向量"""
        if G.number_of_nodes() == 0:
            return torch.zeros(self.motif_dim)
        
        motif_counts = []
        
        # Path-2 motif
        path_2_count = self._count_path_2(G)
        motif_counts.append(path_2_count)
        
        # Fork motif
        fork_count = self._count_fork(G)
        motif_counts.append(fork_count)
        
        # Triangle motif
        triangle_count = self._count_triangles(G)
        motif_counts.append(triangle_count)
        
        # Star motifs
        star_in_count, star_out_count = self._count_stars(G)
        motif_counts.extend([star_in_count, star_out_count])
        
        # Cycle motifs
        cycle_3_count = self._count_cycles(G, 3)
        cycle_4_count = self._count_cycles(G, 4)
        motif_counts.extend([cycle_3_count, cycle_4_count])
        
        # Complete-3 motif
        complete_3_count = self._count_complete_subgraphs(G, 3)
        motif_counts.append(complete_3_count)
        
        # 归一化
        motif_vector = torch.tensor(motif_counts, dtype=torch.float32)
        if motif_vector.sum() > 0:
            motif_vector = motif_vector / (motif_vector.sum() + 1e-8)
        
        return motif_vector
    
    def _count_path_2(self, G: nx.Graph) -> int:
        """计算长度为2的路径数量"""
        count = 0
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    if not G.has_edge(neighbors[i], neighbors[j]):
                        count += 1
        return count
    
    def _count_fork(self, G: nx.Graph) -> int:
        """计算fork结构数量（度>=3的节点）"""
        return sum(1 for node in G.nodes() if G.degree(node) >= 3)
    
    def _count_triangles(self, G: nx.Graph) -> int:
        """计算三角形数量"""
        try:
            triangles = nx.triangles(G)
            return sum(triangles.values()) // 3
        except:
            return 0
    
    def _count_stars(self, G: nx.Graph) -> Tuple[int, int]:
        """计算星形结构数量"""
        if G.is_directed():
            star_in = sum(1 for node in G.nodes() if G.in_degree(node) >= 3)
            star_out = sum(1 for node in G.nodes() if G.out_degree(node) >= 3)
        else:
            star_in = star_out = self._count_fork(G)
        return star_in, star_out
    
    def _count_cycles(self, G: nx.Graph, length: int) -> int:
        """计算指定长度的环数量"""
        try:
            if G.is_directed():
                cycles = list(nx.simple_cycles(G))
                return sum(1 for cycle in cycles if len(cycle) == length)
            else:
                # 对于无向图，使用cycle_basis
                cycles = nx.cycle_basis(G)
                return sum(1 for cycle in cycles if len(cycle) == length)
        except:
            return 0
    
    def _count_complete_subgraphs(self, G: nx.Graph, size: int) -> int:
        """计算完全子图数量"""
        count = 0
        nodes = list(G.nodes())
        from itertools import combinations
        
        for subset in combinations(nodes, size):
            subgraph = G.subgraph(subset)
            if subgraph.number_of_edges() == size * (size - 1) // 2:
                count += 1
        return count