import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import List, Tuple

def nx_to_pyg(G: nx.Graph) -> Data:
    """将NetworkX图转换为PyTorch Geometric数据"""
    # 创建图的副本，移除字符串边属性
    G_clean = nx.Graph()
    
    # 添加节点
    for node in G.nodes():
        G_clean.add_node(node, x=1.0)
    
    # 添加边（不包含字符串属性）
    for u, v in G.edges():
        G_clean.add_edge(u, v)
    
    # 转换为PyTorch Geometric数据
    data = from_networkx(G_clean)
    if data.x is None:
        data.x = torch.ones((data.num_nodes, 1))
    return data

def extract_subgraph(G: nx.Graph, nodes: List, include_edges: bool = True) -> nx.Graph:
    """提取子图"""
    if include_edges:
        return G.subgraph(nodes).copy()
    else:
        subG = nx.Graph()
        subG.add_nodes_from(nodes)
        return subG

def get_neighbors(G: nx.Graph, node, hop: int = 1) -> set:
    """获取k跳邻居"""
    neighbors = {node}
    current = {node}
    
    for _ in range(hop):
        next_neighbors = set()
        for n in current:
            next_neighbors.update(G.neighbors(n))
        current = next_neighbors - neighbors
        neighbors.update(current)
    
    return neighbors - {node}