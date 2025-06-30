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
                          embed_model) -> List[nx.Graph]:
        """从anchor三元组开始扩展子图"""
        h, r, t = anchor
        
        # 初始化子图
        current_subgraph = nx.Graph()
        current_subgraph.add_edge(h, t, relation=r)
        
        # 候选扩展列表 (priority_queue)
        candidates = []
        visited_edges = {(h, t, r)}
        
        for step in range(self.max_steps):
            # 获取当前子图的嵌入
            current_embed = self._get_graph_embedding(current_subgraph, embed_model)
            current_score = self._similarity_score(current_embed, target_embed)
            
            # 找到所有可能的扩展边
            expansion_edges = self._find_expansion_edges(
                current_subgraph, kg, visited_edges
            )
            
            if not expansion_edges:
                break
            
            # 评估每个扩展选项
            best_expansions = []
            for edge in expansion_edges[:self.beam_size * 3]:  # 限制候选数量
                temp_subgraph = current_subgraph.copy()
                eh, et, er = edge
                temp_subgraph.add_edge(eh, et, relation=er)
                
                temp_embed = self._get_graph_embedding(temp_subgraph, embed_model)
                temp_score = self._similarity_score(temp_embed, target_embed)
                
                heapq.heappush(best_expansions, (-temp_score, edge, id(temp_subgraph), temp_subgraph))
            
            # 选择最佳扩展
            if best_expansions:
                _, best_edge, _, best_subgraph = heapq.heappop(best_expansions)
                current_subgraph = best_subgraph
                visited_edges.add(best_edge)
                
                logging.info(f"步骤 {step+1}: 添加边 {best_edge}, 当前相似度: {current_score:.4f}")
            else:
                break
        
        # 返回候选子图列表（这里简化处理，只返回最终扩展的子图）
        return [current_subgraph]
    
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
                return torch.zeros(embed_model.embed_fusion.projection.out_features)
            
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
            # 返回零向量
            return torch.zeros(embed_model.embed_fusion.projection.out_features)
    
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