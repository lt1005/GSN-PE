import torch
import networkx as nx
import numpy as np
from collections import defaultdict
import random
from itertools import combinations

class MotifCounter:
    """计算图中各种motif的出现次数"""
    
    def __init__(self, motif_templates):
        """
        motif_templates: 预定义的motif模板列表，例如:
            [
                ['p1', 'p2'],  # 路径motif
                ['p1', 'p3', 'p2'],  # 三角形motif
                ...
            ]
        """
        self.motif_templates = motif_templates
    
    def count(self, graph):
        """计算图中各种motif的出现次数"""
        # 将图中的边转换为谓词序列
        edge_predicates = [rel for _, rel, _ in graph.edges(data='key')]
        counts = {tuple(motif): 0 for motif in self.motif_templates}
        
        # 对于每个motif模板，检查它在图中的出现次数
        for motif in self.motif_templates:
            motif_str = '-'.join(motif)
            # 生成所有可能的路径并检查是否包含motif
            for path in self._generate_all_paths(graph):
                # 将路径中的元素转换为字符串
                path_str = '-'.join(str(p) for p in path)
                counts[tuple(motif)] += path_str.count(motif_str)
        
        return counts
    
    def _generate_all_paths(self, graph, max_length=5):
        """生成图中所有可能的路径"""
        all_paths = []
        nodes = list(graph.nodes())
        
        # 生成所有可能的节点对
        for source, target in combinations(nodes, 2):
            # 使用networkx的all_simple_paths函数获取所有简单路径
            for path in nx.all_simple_paths(graph, source, target, cutoff=max_length):
                # 将节点路径转换为谓词路径
                predicate_path = []
                for i in range(len(path)-1):
                    # 获取节点间的边谓词
                    edges = graph.get_edge_data(path[i], path[i+1])
                    if edges:
                        for key in edges:
                            predicate_path.append(key)
                if predicate_path:
                    all_paths.append(predicate_path)
        
        return all_paths

class StructureEmbedder:
    """将图结构转换为向量嵌入"""
    
    def __init__(self, motif_counter, embedding_dim=64):
        """
        motif_counter: MotifCounter实例
        embedding_dim: 嵌入向量维度
        """
        self.motif_counter = motif_counter
        self.motif_dim = len(motif_counter.motif_templates)
        self.embedding_dim = embedding_dim
        
        # 初始化权重矩阵，用于将motif计数转换为嵌入向量
        self.weight_matrix = torch.randn(self.motif_dim, embedding_dim)
    
    def embed(self, graph):
        """将图转换为结构嵌入向量"""
        # 计算motif计数
        motif_counts = self.motif_counter.count(graph)
        
        # 将motif计数转换为向量
        count_vector = torch.zeros(self.motif_dim)
        for i, motif in enumerate(self.motif_counter.motif_templates):
            count_vector[i] = motif_counts.get(tuple(motif), 0)
        
        # 应用线性变换得到嵌入向量
        embedding = torch.matmul(count_vector, self.weight_matrix)
        return embedding

class AnchorSubgraphSampler:
    """基于结构嵌入的子图采样器"""
    
    def __init__(self, kg_graph, motif_templates, embedding_dim=64, max_size=10, threshold=0.1):
        """
        kg_graph: 知识图谱 (nx.MultiDiGraph)
        motif_templates: 预定义的motif模板列表
        embedding_dim: 嵌入向量维度
        max_size: 采样子图的最大边数
        threshold: 结构相似度阈值
        """
        self.kg = kg_graph
        self.motif_templates = motif_templates  # e.g. [ [(p1,p2)], [(p3,p4,p5)], ... ]
        self.max_hop = max_hop

    def build(self, anchor_triple):
        head, rel, tail = anchor_triple
        q_nodes = set([head, tail])
        q_edges = set([(head, rel, tail)])

        frontier = set([head, tail])
        visited = set([head, tail])

        for _ in range(self.max_hop):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.kg.neighbors(node):
                    for _, rel_key, edge_data in self.kg.edges(node, neighbor, keys=True, data=True):
                        new_edge = (node, rel_key, neighbor)
                        if self._edge_should_add(q_edges, new_edge):
                            q_edges.add(new_edge)
                            q_nodes.update([node, neighbor])
                            next_frontier.add(neighbor)
            frontier = next_frontier - visited
            visited.update(frontier)

        return self._subgraph_from_edges(q_edges)

    def _edge_should_add(self, current_edges, candidate_edge):
        # 判断加入这条边是否增加motif计数（即 motif 结构支配增强）
        current_path = [rel for (_, rel, _) in current_edges]
        extended_path = current_path + [candidate_edge[1]]
        for motif in self.motif_templates:
            if self._motif_gain(motif, current_path, extended_path):
                return True
        return False

    def _motif_gain(self, motif, current_path, extended_path):
        motif_str = '-'.join(motif)
        path_str_1 = '-'.join(current_path)
        path_str_2 = '-'.join(extended_path)
        return motif_str not in path_str_1 and motif_str in path_str_2

    def _subgraph_from_edges(self, edge_set):
        subg = nx.MultiDiGraph()
        for h, r, t in edge_set:
            subg.add_edge(h, t, key=r)
        return subg


