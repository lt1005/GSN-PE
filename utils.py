# utils.py
import json
import networkx as nx

def load_rules(path):
    """
    读取规则json文件，返回规则列表
    每条规则格式：{"head": ..., "body": [...]}
    """
    with open(path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    return rules

def load_kg(path):
    """
    从edgelist文件加载知识图谱，返回networkx.MultiDiGraph对象
    edgelist格式: head tail relation (以空格分隔)
    """
    G = nx.MultiDiGraph()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                head, tail, rel = parts
                G.add_edge(head, tail, key=rel)
    return G
