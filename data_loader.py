# data_loader.py
import json
import networkx as nx

class KGDataLoader:
    def __init__(self, kg_path):
        """
        加载知识图谱数据，构建NetworkX图结构
        """
        self.kg = nx.MultiDiGraph()
        self._load_kg(kg_path)

    def _load_kg(self, kg_path):
        """
        从边列表文件加载知识图谱，格式：head \t relation \t tail
        """
        with open(kg_path, 'r', encoding='utf-8') as f:
            for line in f:
                h, r, t = line.strip().split('\t')
                self.kg.add_edge(h, t, key=r)

    def get_graph(self):
        return self.kg


class RuleDataLoader:
    def __init__(self, rules_path):
        """
        加载规则数据，格式为json list， 每条规则包含：body（前件），head（后件）
        """
        self.rules = self._load_rules(rules_path)

    def _load_rules(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        return rules

    def get_rules(self):
        return self.rules


class AnchorSubgraphDataset:
    def __init__(self, kg_graph, rules, anchor_builder, max_hop=2):
        """
        基于锚点扩展的子图匹配数据集构建
        参数:
            kg_graph: NetworkX MultiDiGraph，知识图谱
            rules: list，规则集合
            anchor_builder: AnchorSubgraphBuilder实例，用于构建子图
            max_hop: int，扩展最大跳数
        """
        self.kg = kg_graph
        self.rules = rules
        self.anchor_builder = anchor_builder
        self.max_hop = max_hop

    def generate_training_samples(self):
        """
        生成训练样本，返回格式示例：
            [(subgraph1, rule1), (subgraph2, rule2), ...]
        其中subgraph为nx.MultiDiGraph，rule为规则dict
        """
        samples = []
        for rule in self.rules:
            body_triples = rule['body_triples']  # 假设规则包含body_triples键
            for triple in body_triples:
                # 以规则前件中的每个三元组作为锚点，构建子图
                subgraph = self.anchor_builder.build(triple)
                samples.append((subgraph, rule))
        return samples
