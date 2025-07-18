#!/usr/bin/env python3
"""
规则感知的Motif提取器 - 专门针对规则推理优化
"""

import networkx as nx
import torch
import numpy as np
from typing import List, Optional
from collections import defaultdict

class RuleAwareMotifExtractor:
    """专门针对规则推理的motif提取器"""
    
    def __init__(self):
        # 针对规则推理的motif类型
        self.motif_types = [
            'single_edge',           # 单边
            'chain_2',               # 2边链式
            'chain_3',               # 3边链式
            'star_2',                # 2边星形
            'star_3',                # 3边星形
            'triangle',              # 三角形
            'cycle_3',               # 3环
            'cycle_4',               # 4环
            'max_degree',            # 最大度数
            'avg_degree',            # 平均度数
            'density',               # 图密度
            'diameter'               # 图直径
        ]
        self.motif_dim = len(self.motif_types)
    
    def extract_motifs(self, G: nx.Graph) -> torch.Tensor:
        """提取图的motif特征向量"""
        if G.number_of_nodes() == 0:
            return torch.zeros(self.motif_dim)
        
        motif_counts = []
        
        for motif in self.motif_types:
            if motif == 'single_edge':
                motif_counts.append(self._count_single_edges(G))
            elif motif == 'chain_2':
                motif_counts.append(self._count_chain_2(G))
            elif motif == 'chain_3':
                motif_counts.append(self._count_chain_3(G))
            elif motif == 'star_2':
                motif_counts.append(self._count_star_2(G))
            elif motif == 'star_3':
                motif_counts.append(self._count_star_3(G))
            elif motif == 'triangle':
                motif_counts.append(self._count_triangles(G))
            elif motif == 'cycle_3':
                motif_counts.append(self._count_cycles(G, 3))
            elif motif == 'cycle_4':
                motif_counts.append(self._count_cycles(G, 4))
            elif motif == 'max_degree':
                motif_counts.append(self._get_max_degree(G))
            elif motif == 'avg_degree':
                motif_counts.append(self._get_avg_degree(G))
            elif motif == 'density':
                motif_counts.append(self._get_density(G))
            elif motif == 'diameter':
                motif_counts.append(self._get_diameter(G))
            else:
                motif_counts.append(0)
        
        motif_vector = torch.tensor(motif_counts, dtype=torch.float32)
        
        # 归一化处理
        if motif_vector.sum() > 0:
            motif_vector = motif_vector / (motif_vector.sum() + 1e-8)
        
        return motif_vector
    
    def _count_single_edges(self, G: nx.Graph) -> int:
        """计算单边数量"""
        return G.number_of_edges()
    
    def _count_chain_2(self, G: nx.Graph) -> int:
        """计算2边链式结构数量"""
        count = 0
        for node in G.nodes():
            if int(G.degree(node)) == 1:  # 度为1的节点
                count += 1
        return count // 2  # 每条链有两个度为1的节点
    
    def _count_chain_3(self, G: nx.Graph) -> int:
        """计算3边链式结构数量"""
        count = 0
        for node in G.nodes():
            if int(G.degree(node)) == 2:  # 度为2的节点（链的中间节点）
                count += 1
        return count
    
    def _count_star_2(self, G: nx.Graph) -> int:
        """计算2边星形结构数量（一个节点有2条出边）"""
        count = 0
        for node in G.nodes():
            if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
                if int(G.out_degree(node)) == 2:
                    count += 1
            else:
                if int(G.degree(node)) == 2:
                    count += 1
        return count
    
    def _count_star_3(self, G: nx.Graph) -> int:
        """计算3边星形结构数量（一个节点有3条出边）"""
        count = 0
        for node in G.nodes():
            if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
                if int(G.out_degree(node)) >= 3:
                    count += 1
            else:
                if int(G.degree(node)) >= 3:
                    count += 1
        return count
    
    def _count_triangles(self, G: nx.Graph) -> int:
        """计算三角形数量"""
        try:
            if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
                # 转换为无向图计算三角形
                G_undirected = G.to_undirected()
                triangles = nx.triangles(G_undirected)
                return sum(triangles.values()) // 3
            else:
                triangles = nx.triangles(G)
                return sum(triangles.values()) // 3
        except:
            return 0
    
    def _count_cycles(self, G: nx.Graph, length: int) -> int:
        """计算指定长度的环数量"""
        try:
            if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
                cycles = list(nx.simple_cycles(G))
                return sum(1 for cycle in cycles if len(cycle) == length)
            else:
                cycles = nx.cycle_basis(G)
                return sum(1 for cycle in cycles if len(cycle) == length)
        except:
            return 0
    
    def _get_max_degree(self, G: nx.Graph) -> float:
        """获取最大度数"""
        if G.number_of_nodes() == 0:
            return 0.0
        degrees = [G.degree(node) for node in G.nodes()]
        max_deg = max(degrees)
        return float(max_deg) / G.number_of_nodes()  # 归一化
    
    def _get_avg_degree(self, G: nx.Graph) -> float:
        """获取平均度数"""
        if G.number_of_nodes() == 0:
            return 0.0
        degrees = [G.degree(node) for node in G.nodes()]
        total_deg = sum(degrees)
        return float(total_deg) / G.number_of_nodes()
    
    def _get_density(self, G: nx.Graph) -> float:
        """获取图密度"""
        if G.number_of_nodes() <= 1:
            return 0.0
        max_edges = G.number_of_nodes() * (G.number_of_nodes() - 1)
        if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
            max_edges = max_edges
        else:
            max_edges = max_edges // 2
        return float(G.number_of_edges()) / max_edges if max_edges > 0 else 0.0
    
    def _get_diameter(self, G: nx.Graph) -> float:
        """获取图直径"""
        try:
            if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
                G_undirected = G.to_undirected()
                if nx.is_connected(G_undirected):
                    diameter = nx.diameter(G_undirected)
                else:
                    diameter = 0
            else:
                if nx.is_connected(G):
                    diameter = nx.diameter(G)
                else:
                    diameter = 0
            return float(diameter) / max(G.number_of_nodes(), 1)  # 归一化
        except:
            return 0.0

def test_rule_aware_motif():
    """测试规则感知motif提取器"""
    
    print("=== 规则感知Motif测试 ===")
    
    extractor = RuleAwareMotifExtractor()
    
    # 测试图1: 链式结构 A->B->C
    g1 = nx.DiGraph()
    g1.add_edge('A', 'B', relation='r1')
    g1.add_edge('B', 'C', relation='r2')
    
    # 测试图2: 星形结构 A->B, A->C
    g2 = nx.DiGraph()
    g2.add_edge('A', 'B', relation='r1')
    g2.add_edge('A', 'C', relation='r2')
    
    # 测试图3: 三角形 A->B->C->A
    g3 = nx.DiGraph()
    g3.add_edge('A', 'B', relation='r1')
    g3.add_edge('B', 'C', relation='r2')
    g3.add_edge('C', 'A', relation='r3')
    
    test_graphs = [
        ("链式结构", g1),
        ("星形结构", g2),
        ("三角形", g3)
    ]
    
    print("Motif向量对比:")
    for name, graph in test_graphs:
        motif_vec = extractor.extract_motifs(graph)
        print(f"  {name}: {motif_vec.numpy()}")
    
    print("\nMotif差异分析:")
    for i, (name1, graph1) in enumerate(test_graphs):
        motif1 = extractor.extract_motifs(graph1)
        for j, (name2, graph2) in enumerate(test_graphs):
            if i != j:
                motif2 = extractor.extract_motifs(graph2)
                # L1范数
                l1_diff = torch.norm(motif1 - motif2, p=1).item()
                # L2范数
                l2_diff = torch.norm(motif1 - motif2, p=2).item()
                # 余弦相似度
                cos_sim = torch.nn.functional.cosine_similarity(
                    motif1.unsqueeze(0), motif2.unsqueeze(0), dim=1
                ).item()
                
                print(f"  {name1} vs {name2}:")
                print(f"    L1差异: {l1_diff:.4f}")
                print(f"    L2差异: {l2_diff:.4f}")
                print(f"    余弦相似度: {cos_sim:.4f}")

if __name__ == "__main__":
    test_rule_aware_motif() 