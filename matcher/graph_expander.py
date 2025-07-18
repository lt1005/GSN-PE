import networkx as nx
import torch
from typing import List, Tuple, Set, Optional
import heapq
from collections import defaultdict
import logging

class GraphExpander:
    """子图扩展器"""
    
    def __init__(self, max_steps: int = 5, beam_size: int = 3):
        self.max_steps = max_steps
        self.beam_size = beam_size
    
    def expand_from_anchor(self, anchor: Tuple[str, str, str], 
                          kg: nx.MultiDiGraph, 
                          target_embed: torch.Tensor,
                          embed_model,
                          rule_node_count: int = None,
                          rule_edge_count: int = None) -> List[nx.Graph]:
        """beam search结构分驱动扩展，每步保留top-k路径，节点/边数≤规则结构，最终只保留节点/边数=规则结构的子图"""
        import copy
        h, r, t = anchor
        initial_subgraph = nx.Graph()
        initial_subgraph.add_edge(h, t, relation=r)
        visited_edges = {(h, t, r)}
        # beam: (负结构分, 子图, 已访问边)
        beam = [(0, id(initial_subgraph), initial_subgraph, visited_edges)]
        results = []
        for step in range(self.max_steps):
            print(f"[DEBUG] Step {step+1} - beam size: {len(beam)}")
            new_beam = []
            for neg_score, _, subgraph, visited in beam:
                print(f"[DEBUG]  Current subgraph nodes: {list(subgraph.nodes())}, edges: {list(subgraph.edges(data=True))}")
                # 节点/边数超过规则结构则剪枝
                if (rule_node_count and subgraph.number_of_nodes() > rule_node_count) or \
                   (rule_edge_count and subgraph.number_of_edges() > rule_edge_count):
                    print(f"[DEBUG]   Pruned: nodes={subgraph.number_of_nodes()}, edges={subgraph.number_of_edges()}")
                    continue
                # 如果节点/边数正好等于规则结构，加入结果
                if (rule_node_count and subgraph.number_of_nodes() == rule_node_count) and \
                   (rule_edge_count and subgraph.number_of_edges() == rule_edge_count):
                    print(f"[DEBUG]   Candidate result: nodes={subgraph.number_of_nodes()}, edges={subgraph.number_of_edges()}")
                    results.append(subgraph)
                    continue
                expansion_edges = self._find_expansion_edges(subgraph, kg, visited)
                print(f"[DEBUG]   Expansion edges: {expansion_edges}")
                for edge in expansion_edges[:self.beam_size * 3]:
                    eh, et, er = edge
                    if edge in visited:
                        continue
                    temp_subgraph = copy.deepcopy(subgraph)
                    temp_subgraph.add_edge(eh, et, relation=er)
                    temp_embed = self._get_graph_embedding(temp_subgraph, embed_model)
                    temp_score = self._similarity_score(temp_embed, target_embed)
                    print(f"[DEBUG]    New subgraph nodes: {list(temp_subgraph.nodes())}, edges: {list(temp_subgraph.edges(data=True))}, score: {temp_score:.4f}")
                    new_visited = visited | {edge}
                    heapq.heappush(new_beam, (-temp_score, id(temp_subgraph), temp_subgraph, new_visited))
            # 保留top-k
            beam = heapq.nsmallest(self.beam_size, new_beam)
            if not beam:
                break
        print(f"[DEBUG] Final candidate subgraphs before filtering: {len(results)}")
        for g in results:
            print(f"[DEBUG]   Result subgraph nodes: {list(g.nodes())}, edges: {list(g.edges(data=True))}")
        # 最终只保留节点/边数与规则结构一致的子图
        final_results = [g for g in results if \
            (rule_node_count and g.number_of_nodes() == rule_node_count) and \
            (rule_edge_count and g.number_of_edges() == rule_edge_count)]
        print(f"[DEBUG] Final filtered results: {len(final_results)}")
        return final_results
    
    def _find_expansion_edges(self, subgraph: nx.Graph, 
                            kg: nx.MultiDiGraph, 
                            visited: Set) -> List[Tuple[str, str, str]]:
        """找到可能的扩展边"""
        candidates = []
        subgraph_nodes = set(subgraph.nodes())
        
        # 从子图节点出发寻找新边
        for node in subgraph_nodes:
            # 出边
            if node in kg:
                for neighbor in kg.neighbors(node):
                    for edge_data in kg[node][neighbor].values():
                        relation = edge_data.get('relation', 'unknown')
                        edge = (node, neighbor, relation)
                        if edge not in visited:
                            candidates.append(edge)
            
            # 入边
            for pred in kg.predecessors(node):
                for edge_data in kg[pred][node].values():
                    relation = edge_data.get('relation', 'unknown')
                    edge = (pred, node, relation)
                    if edge not in visited:
                        candidates.append(edge)
        
        return candidates
    
    def _get_graph_embedding(self, graph: nx.Graph, embed_model) -> torch.Tensor:
        """获取图的嵌入表示 - 返回融合后的结构嵌入"""
        from utils.graph_utils import nx_to_pyg
        from struct_embed.motif_extractor import MotifExtractor
        
        try:
            # 提取motif特征
            motif_extractor = MotifExtractor()
            motif_embed = motif_extractor.extract_motifs(graph)
            
            # 如果图为空，返回零向量
            if graph.number_of_nodes() == 0:
                out_dim = embed_model.embed_fusion.projection.out_features
                if out_dim is None:
                    out_dim = 40  # fallback默认
                out_dim = int(out_dim)
                return torch.zeros((out_dim,))
            
            # 获取GNN嵌入
            pyg_data = nx_to_pyg(graph)
            pyg_data.batch = torch.zeros(pyg_data.num_nodes, dtype=torch.long)
            
            with torch.no_grad():
                gnn_embed = embed_model.gnn_encoder(pyg_data).squeeze(0)
            
            # 融合嵌入
            if len(motif_embed.shape) == 1:
                motif_embed = motif_embed.unsqueeze(0)
            if len(gnn_embed.shape) == 1:
                gnn_embed = gnn_embed.unsqueeze(0)
            
            fused_embed = embed_model.embed_fusion(motif_embed, gnn_embed)
            return fused_embed.squeeze(0)
        except Exception as e:
            logging.error(f"图嵌入提取失败: {e}")
            out_dim = embed_model.embed_fusion.projection.out_features
            if out_dim is None:
                out_dim = 40  # fallback默认
            out_dim = int(out_dim)
            return torch.zeros((out_dim,))
    
    def _similarity_score(self, embed1: torch.Tensor, embed2: torch.Tensor) -> float:
        """计算两个嵌入向量的相似度"""
        try:
            # 使用余弦相似度
            similarity = torch.nn.functional.cosine_similarity(
                embed1.unsqueeze(0), embed2.unsqueeze(0), dim=1
            )
            return similarity.item()
        except Exception as e:
            logging.error(f"相似度计算失败: {e}")
            return 0.0