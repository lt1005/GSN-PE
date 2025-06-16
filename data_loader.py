import torch
import networkx as nx
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from anchor_expansion import AnchorSubgraphSampler, negative_sample, MotifCounter

def load_relations(relations_file):
    """加载关系字典：关系名称 -> 索引"""
    relation_to_idx = {}
    with open(relations_file, 'r') as f:
        for line in f:
            idx_str, relation = line.strip().split()
            relation_to_idx[relation] = int(idx_str)
    return relation_to_idx

def load_entities(entities_file):
    """加载实体字典：实体名称 -> 索引"""
    entity_to_idx = {}
    with open(entities_file, 'r') as f:
        for line in f:
            idx_str, entity = line.strip().split()
            entity_to_idx[entity] = int(idx_str)
    return entity_to_idx

def load_mined_rules(rules_file):
    """加载挖掘规则：每一行第一个元素为结论，后续为前提"""
    rules = []
    with open(rules_file, 'r') as f:
        for line in f:
            parts = list(map(int, line.strip().split()))
            if len(parts) > 1:
                rule_head = parts[0]
                rule_body = parts[1:]
                rules.append([rule_head, rule_body])
    return rules

def load_triple_data(triple_file):
    """加载三元组数据：头实体 关系 尾实体"""
    triples = []
    with open(triple_file, 'r') as f:
        for line in f:
            head, rel, tail = line.strip().split()
            triples.append((head, rel, tail))
    return triples

class RuleDataset(Dataset):
    """规则数据集，处理规则、知识图谱、子图采样"""
    
    def __init__(self, rules_file, kg_files, motif_templates, predicate_vocab, entity_vocab,
                 num_negatives=1, max_subgraph_size=10, transform=None):
        self.rules = load_mined_rules(rules_file)
        # 明确赋值
        self.predicate_vocab = predicate_vocab  # 关系名到索引的映射
        self.entity_vocab = entity_vocab      # 实体名到索引的映射
        self.kg = self._load_kg(kg_files)
        self.motif_templates = motif_templates
        self.num_negatives = num_negatives
        self.max_subgraph_size = max_subgraph_size
        self.transform = transform
        
        # 初始化子图采样器和Motif计数器
        self.sampler = AnchorSubgraphSampler(
            self.kg, motif_templates, max_size=max_subgraph_size
        )
        self.motif_counter = MotifCounter(motif_templates)
    
    def _load_kg(self, kg_files):
        """从多个KG文件加载知识图谱（支持train/valid/test.txt）"""
        kg = nx.MultiDiGraph()
        for file in kg_files:
            with open(file, 'r') as f:
                for line in f:
                    head, rel, tail = line.strip().split()
                    # 转换为索引后添加到图中
                    if head in self.entity_vocab and rel in self.predicate_vocab and tail in self.entity_vocab:
                        kg.add_edge(
                            self.entity_vocab[head], 
                            self.entity_vocab[tail], 
                            key=self.predicate_vocab[rel]
                        )
        return kg
    
    def __len__(self):
        return len(self.rules)
    
    def __getitem__(self, idx):
        """获取样本：规则、正负子图、Motif计数"""
        rule_head, premises = self.rules[idx]
        premise_indices = [self.predicate_vocab.get(p, 0) for p in premises]  # 前提关系索引
        
        # 构建规则图并计算Motif
        rule_graph = self.sampler.build_rule_graph(premises)
        premise_motif = self._get_motif_counts(rule_graph)
        
        # 采样正负子图
        pos_subgraph = self.sampler.sample(premises)
        pos_motif = self._get_motif_counts(pos_subgraph)
        
        neg_subgraph = negative_sample(
            premises, self.kg, self.motif_templates, max_size=self.max_subgraph_size
        )
        neg_motif = self._get_motif_counts(neg_subgraph)
        
        return {
            'rule_head': torch.tensor(rule_head, dtype=torch.long),
            'premises': torch.tensor(premise_indices, dtype=torch.long),
            'premise_motif': premise_motif,
            'pos_subgraph': pos_subgraph,
            'pos_motif': pos_motif,
            'neg_subgraph': neg_subgraph,
            'neg_motif': neg_motif
        }
    
    def _get_motif_counts(self, graph):
        """计算图中各Motif的出现次数"""
        if graph is None or graph.number_of_edges() == 0:
            return torch.zeros(len(self.motif_templates))
        counts = self.motif_counter.count(graph)
        return torch.tensor([counts.get(tuple(motif), 0) for motif in self.motif_templates])

def collate_fn(batch):
    """批处理函数：处理不同长度的规则和子图"""
    rule_heads = torch.tensor([item['rule_head'] for item in batch], dtype=torch.long)
    premises = [item['premises'] for item in batch]
    premise_motif = torch.stack([item['premise_motif'] for item in batch])
    pos_motif = torch.stack([item['pos_motif'] for item in batch])
    neg_motif = torch.stack([item['neg_motif'] for item in batch])
    
    return {
        'rule_heads': rule_heads,
        'premises': premises,
        'premise_motif': premise_motif,
        'pos_motif': pos_motif,
        'neg_motif': neg_motif
    }
