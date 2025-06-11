# anchor_expansion.py
# Step 1: Anchor Predicate Expansion (构建结构完整子图)
# 输入：锚定三元组、KG邻接表、motif模板库；输出：子图Q

import networkx as nx
from collections import defaultdict

class AnchorSubgraphBuilder:
    def __init__(self, kg_graph: nx.MultiDiGraph, motif_templates: list, max_hop: int = 2):
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


