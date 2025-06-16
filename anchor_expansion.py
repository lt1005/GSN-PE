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
        self.motif_counter = MotifCounter(motif_templates)
        self.embedder = StructureEmbedder(self.motif_counter, embedding_dim)
        self.max_size = max_size
        self.threshold = threshold
    
    def build_rule_graph(self, rule_predicates):
        """构建规则图（无实体，仅谓词节点及边）"""
        rule_graph = nx.MultiDiGraph()
        
        # 添加谓词节点和边
        for i in range(len(rule_predicates)-1):
            rule_graph.add_edge(rule_predicates[i], rule_predicates[i+1], key=rule_predicates[i])
        
        return rule_graph
    
    def sample(self, rule_predicates, num_samples=10):
        """
        从知识图谱中采样与规则匹配的子图
        
        参数:
            rule_predicates: 规则前件谓词集合 [p1, p2, ..., pm]
            num_samples: 采样的子图数量
        
        返回:
            与规则匹配的子图列表
        """
        # 构建规则图并计算其结构嵌入
        rule_graph = self.build_rule_graph(rule_predicates)
        rule_embedding = self.embedder.embed(rule_graph)
        
        # 查找知识图谱中包含规则谓词的所有三元组
        candidate_triples = []
        for h, t, key in self.kg.edges(data='key'):
            if key in rule_predicates:
                candidate_triples.append((h, key, t))
        
        # 采样子图
        sampled_subgraphs = []
        for _ in range(num_samples):
            # 随机选择一个候选三元组作为起点
            if not candidate_triples:
                continue
                
            start_triple = random.choice(candidate_triples)
            subgraph = self._expand_from_triple(start_triple, rule_predicates, rule_embedding)
            if subgraph:
                sampled_subgraphs.append(subgraph)
        
        # 选择与规则最匹配的子图
        if sampled_subgraphs:
            best_subgraph = self._select_best_subgraph(sampled_subgraphs, rule_embedding)
            return best_subgraph
        return None
    
    def _expand_from_triple(self, start_triple, rule_predicates, rule_embedding):
        """从起始三元组开始扩展子图"""
        head, rel, tail = start_triple
        subgraph = nx.MultiDiGraph()
        subgraph.add_edge(head, tail, key=rel)
        
        # 计算当前子图的结构嵌入
        current_embedding = self.embedder.embed(subgraph)
        current_distance = torch.norm(current_embedding - rule_embedding).item()
        
        # 递归扩展子图
        for _ in range(self.max_size - 1):
            # 获取所有可能的扩展边
            candidate_edges = self._get_candidate_edges(subgraph, rule_predicates)
            if not candidate_edges:
                break
                
            # 计算每条候选边的扩展增益
            best_edge = None
            best_gain = -float('inf')
            
            for edge in candidate_edges:
                # 临时添加边
                temp_subgraph = subgraph.copy()
                temp_subgraph.add_edge(edge[0], edge[2], key=edge[1])
                
                # 计算扩展后的结构嵌入和距离
                new_embedding = self.embedder.embed(temp_subgraph)
                new_distance = torch.norm(new_embedding - rule_embedding).item()
                
                # 计算增益
                gain = current_distance - new_distance
                
                if gain > best_gain and gain > self.threshold:
                    best_gain = gain
                    best_edge = edge
            
            # 如果没有找到有增益的边，停止扩展
            if best_edge is None:
                break
                
            # 添加最佳边
            subgraph.add_edge(best_edge[0], best_edge[2], key=best_edge[1])
            current_embedding = self.embedder.embed(subgraph)
            current_distance = torch.norm(current_embedding - rule_embedding).item()
            
            # 检查是否包含所有规则谓词
            subgraph_predicates = set([rel for _, rel, _ in subgraph.edges(data='key')])
            if subgraph_predicates.issuperset(set(rule_predicates)):
                break
        
        return subgraph
    
    def _get_candidate_edges(self, subgraph, rule_predicates):
        """获取子图的候选扩展边"""
        candidate_edges = []
        
        # 获取子图的所有节点
        nodes = list(subgraph.nodes())
        
        # 对于每个节点，查找其在知识图谱中的邻居
        for node in nodes:
            for neighbor in self.kg.neighbors(node):
                for _, rel_key, _ in self.kg.edges(node, neighbor, keys=True, data=True):
                    # 如果边的谓词在规则谓词集合中，且边不在当前子图中
                    if rel_key in rule_predicates and (node, rel_key, neighbor) not in subgraph.edges(data='key'):
                        candidate_edges.append((node, rel_key, neighbor))
        
        return candidate_edges
    
    def _select_best_subgraph(self, subgraphs, rule_embedding):
        """选择与规则最匹配的子图"""
        best_subgraph = None
        best_distance = float('inf')
        
        for subgraph in subgraphs:
            embedding = self.embedder.embed(subgraph)
            distance = torch.norm(embedding - rule_embedding).item()
            
            if distance < best_distance:
                best_distance = distance
                best_subgraph = subgraph
        
        return best_subgraph

def negative_sample(rule_predicates, kg_graph, motif_templates, embedding_dim=64, max_size=10, threshold=0.1):
    """生成负样本（与规则不匹配的子图）"""
    # 随机扰动规则谓词集
    new_predicates = rule_predicates.copy()
    operation = random.choice(['replace', 'add', 'delete'])
    
    if operation == 'replace' and len(new_predicates) > 0:
        # 替换一个谓词
        index = random.randint(0, len(new_predicates) - 1)
        all_predicates = set([rel for _, rel, _ in kg_graph.edges(data='key')])
        available_predicates = all_predicates - set(new_predicates)
        if available_predicates:
            new_predicates[index] = random.choice(list(available_predicates))
    
    elif operation == 'add':
        # 添加一个谓词
        all_predicates = set([rel for _, rel, _ in kg_graph.edges(data='key')])
        available_predicates = all_predicates - set(new_predicates)
        if available_predicates:
            new_predicates.append(random.choice(list(available_predicates)))
    
    elif operation == 'delete' and len(new_predicates) > 1:
        # 删除一个谓词
        index = random.randint(0, len(new_predicates) - 1)
        del new_predicates[index]
    
    # 使用扰动后的谓词集采样子图
    sampler = AnchorSubgraphSampler(kg_graph, motif_templates, embedding_dim, max_size, threshold)
    return sampler.sample(new_predicates)