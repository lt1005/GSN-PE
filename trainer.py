import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
import os

class RuleDataset(Dataset):
    """规则与子图对齐数据集"""
    
    def __init__(self, rules_file, kg_files, motif_templates, predicate_vocab, entity_vocab, max_subgraph_size=8):
        """
        初始化数据集
        
        参数:
            rules_file: 规则文件路径
            kg_files: 知识图谱文件列表
            motif_templates: motif模板列表
            predicate_vocab: 关系词汇表
            entity_vocab: 实体词汇表
            max_subgraph_size: 最大子图大小
        """
        self.rules = self._load_rules(rules_file)
        self.kg = self._load_kg(kg_files)
        self.motif_templates = motif_templates
        self.predicate_vocab = predicate_vocab
        self.entity_vocab = entity_vocab
        self.max_subgraph_size = max_subgraph_size
        
        # 构建实体和关系到索引的映射
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(entity_vocab)}
        self.predicate_to_idx = {predicate: idx for idx, predicate in enumerate(predicate_vocab)}
        
        # 预处理规则和知识图谱
        self._preprocess()
    
    def _load_rules(self, rules_file):
        """加载规则"""
        rules = []
        with open(rules_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 解析规则：前提 -> 结论
                premise, conclusion = line.split('->')
                premises = [p.strip() for p in premise.split(',')]
                conclusion = conclusion.strip()
                rules.append({'premises': premises, 'conclusion': conclusion})
        return rules
    
    def _load_kg(self, kg_files):
        """加载知识图谱"""
        kg = nx.MultiDiGraph()
        for kg_file in kg_files:
            with open(kg_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    head, relation, tail = line.split()
                    kg.add_edge(head, tail, key=relation, relation=relation)
        return kg
    
    def _preprocess(self):
        """预处理规则和知识图谱"""
        # 可以添加一些预处理步骤，如构建索引等
        pass
    
    def _extract_subgraph(self, entity, radius=2):
        """
        从知识图谱中提取以entity为中心的子图
        
        参数:
            entity: 中心实体
            radius: 半径，即跳数
        
        返回:
            subgraph: 提取的子图
        """
        # 使用BFS提取子图
        visited = set([entity])
        queue = [(entity, 0)]
        
        while queue:
            current_entity, dist = queue.pop(0)
            if dist >= radius:
                continue
            
            # 处理出边
            for neighbor in self.kg.successors(current_entity):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
            
            # 处理入边
            for neighbor in self.kg.predecessors(current_entity):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        
        # 提取子图
        subgraph = self.kg.subgraph(visited)
        return subgraph
    
    def _count_motifs(self, graph, motif_templates):
        """
        计算图中各种motif的出现次数
        
        参数:
            graph: 输入图
            motif_templates: motif模板列表
        
        返回:
            motif_counts: 各种motif的出现次数
        """
        motif_counts = np.zeros(len(motif_templates), dtype=np.float32)
        
        # 这里简化处理，实际应用中可能需要更复杂的motif检测算法
        for i, template in enumerate(motif_templates):
            # 示例：简单统计特定类型的边
            if len(template) == 2 and template[0] == 'p1' and template[1] == 'p2':
                # 统计有两个不同关系的路径
                count = 0
                for u, v, data in graph.edges(data=True):
                    # 这里只是简单示例，实际应用中需要更复杂的匹配
                    count += 1
                motif_counts[i] = count
        
        # 归一化处理
        if np.sum(motif_counts) > 0:
            motif_counts = motif_counts / np.sum(motif_counts)
        
        return torch.tensor(motif_counts, dtype=torch.float32)
    
    def _parse_rule(self, rule):
        """
        解析规则，将实体和关系转换为索引
        
        参数:
            rule: 规则字典，包含premises和conclusion
        
        返回:
            parsed_rule: 解析后的规则
        """
        parsed_premises = []
        for premise in rule['premises']:
            head, relation, tail = premise.split()
            # 将实体和关系转换为索引
            head_idx = self.entity_to_idx.get(head, -1)
            relation_idx = self.predicate_to_idx.get(relation, -1)
            tail_idx = self.entity_to_idx.get(tail, -1)
            
            if head_idx == -1 or relation_idx == -1 or tail_idx == -1:
                # 跳过不在词汇表中的实体或关系
                continue
            
            parsed_premises.append((head_idx, relation_idx, tail_idx))
        
        return parsed_premises
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.rules)
    
    def __getitem__(self, idx):
        """获取一个训练样本"""
        rule = self.rules[idx]
        
        # 解析规则
        premises = self._parse_rule(rule)
        if not premises:
            # 如果解析后的前提为空，返回一个随机样本或跳过
            return self.__getitem__(np.random.randint(0, len(self.rules)))
        
        # 提取正样本子图（与规则匹配的子图）
        conclusion = rule['conclusion']
        head, relation, tail = conclusion.split()
        
        # 确保结论中的实体在知识图谱中
        if head not in self.kg or tail not in self.kg:
            return self.__getitem__(np.random.randint(0, len(self.rules)))
        
        # 提取正样本子图
        positive_subgraph = self._extract_subgraph(head)
        
        # 提取负样本子图（随机子图或不匹配的子图）
        # 这里简化处理，随机选择一个实体
        random_entity = np.random.choice(list(self.kg.nodes()))
        negative_subgraph = self._extract_subgraph(random_entity)
        
        # 计算motif计数
        premise_motif_counts = self._count_motifs(positive_subgraph, self.motif_templates)
        positive_motif_counts = self._count_motifs(positive_subgraph, self.motif_templates)
        negative_motif_counts = self._count_motifs(negative_subgraph, self.motif_templates)
        
        return {
            'premises': premises,
            'premise_motif_counts': premise_motif_counts,
            'positive_subgraph': positive_subgraph,
            'positive_motif_counts': positive_motif_counts,
            'negative_subgraph': negative_subgraph,
            'negative_motif_counts': negative_motif_counts
        }