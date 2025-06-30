import os
import sys
import torch
import logging
import networkx as nx
from typing import Dict, List, Tuple, Any
import json

# 导入所有模块
from utils.config import Config
from utils.logging_utils import setup_logging
from utils.graph_utils import nx_to_pyg, extract_subgraph
from data.kg_loader import KGLoader
from data.rule_loader import RuleLoader
from struct_embed.motif_extractor import MotifExtractor
from struct_embed.gnn_encoder import GNNEncoder
from struct_embed.embed_fusion import EmbedFusion
from matcher.graph_expander import GraphExpander
from matcher.struct_matcher import StructMatcher
from matcher.scoring import ScoringFunction
from trainer.train_gnn import GNNTrainer
from trainer.contrastive_loss import ContrastiveLoss

class StructMatchSystem:
    """StructMatch主系统"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging()
        
        # 初始化组件
        self.motif_extractor = MotifExtractor(config.motif_dim)
        self.gnn_encoder = GNNEncoder(
            input_dim=1,
            hidden_dim=config.gnn_hidden_dim,
            output_dim=config.gnn_output_dim
        )
        self.embed_fusion = EmbedFusion(
            motif_dim=config.motif_dim,
            gnn_dim=config.gnn_output_dim,
            fusion_dim=config.fusion_dim
        )
        self.graph_expander = GraphExpander(
            max_steps=config.max_expand_steps,
            beam_size=config.beam_size
        )
        self.struct_matcher = StructMatcher(
            threshold=config.similarity_threshold
        )
        
        # 知识图谱和规则
        self.kg = None
        self.rules = None
        
        self.logger.info("StructMatch系统初始化完成")
    
    def load_data(self, kg_path: str, rules_path: str):
        """加载知识图谱和规则数据"""
        self.logger.info("开始加载数据...")
        
        # 加载知识图谱
        kg_loader = KGLoader()
        self.kg = kg_loader.load_kg(kg_path)
        
        # 加载规则
        rule_loader = RuleLoader()
        self.rules = rule_loader.load_rules(rules_path)
        
        self.logger.info("数据加载完成")
    
    def get_structure_embedding(self, graph: nx.Graph) -> torch.Tensor:
        """获取图的结构嵌入"""
        # 提取motif特征
        motif_embed = self.motif_extractor.extract_motifs(graph)
        
        # 获取GNN嵌入
        if graph.number_of_nodes() > 0:
            pyg_data = nx_to_pyg(graph)
            pyg_data.batch = torch.zeros(pyg_data.num_nodes, dtype=torch.long)
            
            with torch.no_grad():
                gnn_embed = self.gnn_encoder(pyg_data).squeeze(0)
        else:
            gnn_embed = torch.zeros(self.config.gnn_output_dim)
        
        # 融合嵌入
        if len(motif_embed.shape) == 1:
            motif_embed = motif_embed.unsqueeze(0)
        if len(gnn_embed.shape) == 1:
            gnn_embed = gnn_embed.unsqueeze(0)
            
        fused_embed = self.embed_fusion(motif_embed, gnn_embed)
        return fused_embed.squeeze(0)
    
    def create_rule_graph(self, rule: Dict) -> nx.Graph:
        """根据规则创建图结构"""
        rule_graph = nx.Graph()
        
        # 解析规则结构
        if 'premise' in rule:
            premise = rule['premise']
            if isinstance(premise, list):
                # 处理三元组列表
                for triple in premise:
                    if len(triple) == 3:
                        h, r, t = triple
                        rule_graph.add_edge(h, t, relation=r)
            elif isinstance(premise, dict):
                # 处理结构化规则描述
                if 'structure_type' in premise:
                    rule_graph = self._create_structure_by_type(premise)
        
        return rule_graph
    
    def _create_structure_by_type(self, structure_desc: Dict) -> nx.Graph:
        """根据结构类型创建图"""
        graph = nx.Graph()
        struct_type = structure_desc.get('structure_type', 'unknown')
        
        if struct_type == 'T_shape':
            # T型结构: A-B-C, B-D
            graph.add_edges_from([('A', 'B'), ('B', 'C'), ('B', 'D')])
        elif struct_type == 'Y_shape':
            # Y型结构: A-D, B-D, C-D
            graph.add_edges_from([('A', 'D'), ('B', 'D'), ('C', 'D')])
        elif struct_type == 'Fork':
            # Fork结构: A-B, A-C, A-D
            graph.add_edges_from([('A', 'B'), ('A', 'C'), ('A', 'D')])
        elif struct_type == 'Triangle':
            # 三角形: A-B-C-A
            graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        elif struct_type == 'Path':
            # 路径结构
            length = structure_desc.get('length', 3)
            nodes = [f'N{i}' for i in range(length)]
            edges = [(nodes[i], nodes[i+1]) for i in range(length-1)]
            graph.add_edges_from(edges)
        
        return graph
    
    def match_subgraph_to_rule(self, anchor: Tuple[str, str, str], 
                              rule: Dict) -> Dict[str, Any]:
        """将anchor扩展并匹配到规则"""
        self.logger.info(f"开始处理anchor: {anchor}")
        
        # 创建规则图
        rule_graph = self.create_rule_graph(rule)
        if rule_graph.number_of_nodes() == 0:
            self.logger.warning(f"无法创建规则图: {rule}")
            return {'success': False, 'reason': 'Invalid rule structure'}
        
        # 获取规则嵌入
        rule_embed = self.get_structure_embedding(rule_graph)
        
        # 扩展子图
        try:
            expanded_subgraph = self.graph_expander.expand_from_anchor(
                anchor, self.kg, rule_embed, self
            )
        except Exception as e:
            self.logger.error(f"子图扩展失败: {e}")
            return {'success': False, 'reason': f'Expansion failed: {e}'}
        
        # 获取扩展子图的嵌入
        subgraph_embed = self.get_structure_embedding(expanded_subgraph)
        
        # 结构匹配
        is_match = self.struct_matcher.is_match(subgraph_embed, rule_embed)
        match_score = self.struct_matcher.match_score(subgraph_embed, rule_embed)
        
        result = {
            'success': True,
            'anchor': anchor,
            'rule': rule,
            'expanded_subgraph': expanded_subgraph,
            'is_match': is_match,
            'match_score': match_score,
            'subgraph_nodes': list(expanded_subgraph.nodes()),
            'subgraph_edges': list(expanded_subgraph.edges()),
            'rule_graph_info': {
                'nodes': list(rule_graph.nodes()),
                'edges': list(rule_graph.edges())
            }
        }
        
        self.logger.info(f"匹配结果: {is_match}, 得分: {match_score:.4f}")
        return result
    
    def run_batch_matching(self, anchors: List[Tuple[str, str, str]], 
                          rule_idx: int = 0) -> List[Dict[str, Any]]:
        """批量处理anchor匹配"""
        if not self.rules or rule_idx >= len(self.rules):
            raise ValueError("Invalid rule index or no rules loaded")
        
        target_rule = self.rules[rule_idx]
        results = []
        
        self.logger.info(f"开始批量匹配，共{len(anchors)}个anchor")
        
        for i, anchor in enumerate(anchors):
            self.logger.info(f"处理第{i+1}/{len(anchors)}个anchor")
            result = self.match_subgraph_to_rule(anchor, target_rule)
            results.append(result)
        
        # 统计结果
        successful_matches = sum(1 for r in results if r.get('is_match', False))
        self.logger.info(f"批量匹配完成: {successful_matches}/{len(anchors)} 成功匹配")
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str):
        """保存结果到文件"""
        import json
        
        # 处理不可序列化的对象
        serializable_results = []
        for result in results:
            serializable_result = result.copy()
            
            # 转换NetworkX图为边列表
            if 'expanded_subgraph' in serializable_result:
                graph = serializable_result['expanded_subgraph']
                serializable_result['expanded_subgraph'] = {
                    'nodes': list(graph.nodes()),
                    'edges': [(u, v, graph[u][v]) for u, v in graph.edges()]
                }
            
            serializable_results.append(serializable_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存到: {output_path}")

def create_sample_data():
    """创建示例数据文件"""
    # 创建示例知识图谱
    kg_data = """A	friendOf	B
B	friendOf	C
C	friendOf	D
A	worksWith	C
B	worksWith	D
A	livesIn	Tokyo
B	livesIn	Tokyo
C	livesIn	Osaka
D	livesIn	Osaka
E	friendOf	F
F	friendOf	G
E	worksWith	G"""
    
    # 创建示例规则
    rules_data = [
        {
            "id": 1,
            "name": "Friendship Triangle",
            "premise": {
                "structure_type": "Triangle",
                "description": "三角形友谊关系"
            },
            "conclusion": "Strong social bond"
        },
        {
            "id": 2,
            "name": "Work-Friend Fork",
            "premise": {
                "structure_type": "Fork",
                "description": "工作和友谊的分叉结构"
            },
            "conclusion": "Professional and personal connection"
        },
        {
            "id": 3,
            "name": "T-shaped Network",
            "premise": {
                "structure_type": "T_shape",
                "description": "T型网络结构"
            },
            "conclusion": "Central connector role"
        }
    ]
    
    # 创建目录
    os.makedirs("data", exist_ok=True)
    
    # 写入文件
    with open("data/example_kg.txt", "w", encoding="utf-8") as f:
        f.write(kg_data)
    
    with open("data/rules.json", "w", encoding="utf-8") as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)
    
    print("示例数据文件已创建:")
    print("- data/example_kg.txt")
    print("- data/rules.json")

def main():
    """主函数示例"""
    # 设置配置
    config = Config()
    
    # 创建示例数据
    create_sample_data()
    
    # 初始化系统
    system = StructMatchSystem(config)
    
    # 加载数据
    try:
        system.load_data("data/example_kg.txt", "data/rules.json")
    except FileNotFoundError as e:
        print(f"数据文件未找到: {e}")
        print("请确保存在 data/example_kg.txt 和 data/rules.json 文件")
        return
    
    # 定义测试anchor
    test_anchors = [
        ("A", "friendOf", "B"),
        ("B", "worksWith", "D"),
        ("E", "friendOf", "F"),
        ("A", "livesIn", "Tokyo")
    ]
    
    # 运行匹配
    print("\n开始结构匹配...")
    results = system.run_batch_matching(test_anchors, rule_idx=0)
    
    # 显示结果
    print("\n=== 匹配结果 ===")
    for i, result in enumerate(results):
        if result['success']:
            print(f"\nAnchor {i+1}: {result['anchor']}")
            print(f"匹配成功: {result['is_match']}")
            print(f"匹配得分: {result['match_score']:.4f}")
            print(f"扩展子图节点: {result['subgraph_nodes']}")
            print(f"扩展子图边: {result['subgraph_edges']}")
        else:
            if 'anchor' in result:
                print(f"\nAnchor {i+1}: {result['anchor']} - 处理失败")
            else:
                print(f"\nAnchor {i+1}: 未知anchor - 处理失败，result内容: {result}")
    
    # 保存结果
    system.save_results(results, "results.json")
    print(f"\n详细结果已保存到 results.json")

if __name__ == "__main__":
    main()
