import os
import sys
import torch
import logging
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import json
import time
from dataclasses import dataclass
import argparse
from collections import defaultdict
import hashlib

# 导入所有模块
from utils.config import Config
from utils.logging_utils import setup_logging
from utils.graph_utils import nx_to_pyg, extract_subgraph
from data.kg_loader import KGLoader
from data.rule_loader import RuleLoader
# 导入增强的结构嵌入模块
from struct_embed.motif_extractor import MotifExtractor
from struct_embed.gnn_encoder import GNNEncoder
from struct_embed.embed_fusion import EmbedFusion
from matcher.graph_expander import GraphExpander
from matcher.struct_matcher import StructMatcher
from matcher.scoring import ScoringFunction
from matcher.semantic_verifier import SemanticVerifier
from trainer.train_gnn import GNNTrainer
from trainer.contrastive_loss import ContrastiveLoss
# 导入语义增强模块
from semantic_enhancer import SemanticFeatureEnhancer, EnhancedSemanticVerifier

@dataclass
class MatchingResult:
    """匹配结果数据结构"""
    rule_id: str
    anchor: Tuple[str, str, str]
    matched_subgraphs: List[Dict]
    structure_score: float
    semantic_score: float
    final_score: float
    is_matched: bool
    processing_time: float

@dataclass
class EvaluationMetrics:
    """评估指标数据结构"""
    structure_accuracy: float = 0.0
    isomorphic_accuracy: float = 0.0
    semantic_accuracy: float = 0.0
    e2e_accuracy: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    avg_processing_time: float = 0.0

class StructMatchSystem:
    """StructMatch主系统 - 基于SRS重构，集成语义增强"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging()
        
        # 初始化语义增强模块
        self.semantic_enhancer = SemanticFeatureEnhancer(
            embedding_dim=config.embedding_dim,
            feature_dim=config.semantic_feature_dim
        )
        
        # 初始化结构嵌入组件
        self.motif_extractor = MotifExtractor()
        self.gnn_encoder = GNNEncoder(
            input_dim=1,
            hidden_dim=config.gnn_hidden_dim,
            output_dim=config.gnn_output_dim,
            gnn_type="GIN",
            num_layers=3
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
        self.scoring_function = ScoringFunction()
        
        # 使用增强的语义验证器
        self.semantic_verifier = EnhancedSemanticVerifier(
            semantic_enhancer=self.semantic_enhancer,
            threshold=config.semantic_threshold,
            feature_dim=config.semantic_feature_dim
        )
        
        # 知识图谱和规则
        self.kg = None
        self.rules = None
        
        # 评估指标
        self.evaluation_metrics = EvaluationMetrics()
        
        self.logger.info("StructMatch系统初始化完成（集成语义增强）")
    
    def load_data(self, kg_path: str, rules_path: str):
        """加载知识图谱和规则数据"""
        self.logger.info("开始加载数据...")
        
        try:
            # 加载知识图谱
            kg_loader = KGLoader()
            self.kg = kg_loader.load_kg(kg_path)
            
            # 加载规则
            rule_loader = RuleLoader()
            self.rules = rule_loader.load_rules(rules_path)
            
            self.logger.info("数据加载完成")
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            raise
    
    def collect_anchors_for_rule(self, rule: Dict) -> List[Tuple[str, str, str]]:
        """收集规则第一个谓词对应的anchor"""
        anchors = []
        
        if not rule.get('premise'):
            return anchors
        
        # 获取规则第一个谓词的关系类型
        first_triple = rule['premise'][0]
        # 处理数组格式的规则 [h, r, t]
        if isinstance(first_triple, list):
            first_relation = first_triple[1]  # 数组格式：["X", "_also_see", "Y"]
        else:
            first_relation = first_triple['r']  # 字典格式：{"h": "X", "r": "_also_see", "t": "Y"}
        
        self.logger.info(f"收集规则第一个谓词对应的anchor: {first_relation}")
        
        # 检查知识图谱是否已加载
        if self.kg is None:
            self.logger.error("知识图谱未加载")
            return anchors
        
        # 从知识图谱中收集对应的三元组
        for h, t, data in self.kg.edges(data=True):
            relation = data.get('relation', '')
            if relation == first_relation:
                anchors.append((h, relation, t))
        
        self.logger.info(f"收集到 {len(anchors)} 个anchor")
        return anchors
    
    def get_structure_embedding(self, graph: nx.Graph) -> torch.Tensor:
        """获取图的结构嵌入表示"""
        try:
            # 提取motif特征
            motif_embed = self.motif_extractor.extract_motifs(graph)
            
            # 如果图为空，返回零向量
            if graph.number_of_nodes() == 0:
                out_dim = getattr(self.embed_fusion.projection, 'out_features', 40)
                if out_dim is None or not isinstance(out_dim, int):
                    out_dim = 40
                return torch.zeros((int(out_dim),))
            
            # 获取GNN嵌入
            pyg_data = nx_to_pyg(graph)
            if pyg_data.num_nodes is not None:
                pyg_data.batch = torch.zeros(pyg_data.num_nodes, dtype=torch.long)
            else:
                pyg_data.batch = torch.zeros(0, dtype=torch.long)
            
            with torch.no_grad():
                gnn_embed = self.gnn_encoder(pyg_data).squeeze(0)
            
            # 融合嵌入
            if len(motif_embed.shape) == 1:
                motif_embed = motif_embed.unsqueeze(0)
            if len(gnn_embed.shape) == 1:
                gnn_embed = gnn_embed.unsqueeze(0)
            
            fused_embed = self.embed_fusion(motif_embed, gnn_embed)
            return fused_embed.squeeze(0)
        except Exception as e:
            self.logger.error(f"结构嵌入提取失败: {e}")
            out_dim = getattr(self.embed_fusion.projection, 'out_features', 40)
            if out_dim is None or not isinstance(out_dim, int):
                out_dim = 40
            return torch.zeros((int(out_dim),))
    
    def get_semantic_embedding(self, graph: nx.Graph) -> torch.Tensor:
        """获取图的语义嵌入"""
        try:
            # 使用语义增强器获取语义特征
            return self.semantic_enhancer.get_subgraph_semantic_features(graph)
        except Exception as e:
            self.logger.error(f"语义嵌入提取失败: {e}")
            out_dim = getattr(self.config, 'semantic_feature_dim', 40)
            if out_dim is None or not isinstance(out_dim, int):
                out_dim = 40
            return torch.zeros((int(out_dim),))
    
    def precompute_rule_embeddings(self, rule: Dict) -> Tuple[torch.Tensor, torch.Tensor, nx.Graph]:
        """预计算规则的结构嵌入和语义嵌入，并返回规则结构图"""
        try:
            # 构建规则前件图
            rule_graph = nx.Graph()
            for triple in rule['premise']:
                # 处理数组格式的规则 [h, r, t]
                if isinstance(triple, list):
                    h, r, t = triple[0], triple[1], triple[2]
                else:
                    h, r, t = triple['h'], triple['r'], triple['t']
                rule_graph.add_edge(h, t, relation=r)
            # 计算结构嵌入
            struct_embed = self.get_structure_embedding(rule_graph)
            # 计算语义嵌入
            semantic_embed = self.get_semantic_embedding(rule_graph)
            return struct_embed, semantic_embed, rule_graph
        except Exception as e:
            self.logger.error(f"规则嵌入预计算失败: {e}")
            raise
    
    def match_subgraph_to_rule(self, anchor: Tuple[str, str, str], 
                              target_rule: Dict) -> MatchingResult:
        from networkx.algorithms import isomorphism
        start_time = time.time()
        try:
            # 预计算规则嵌入和结构图
            rule_struct_embed, rule_semantic_embed, rule_graph = self.precompute_rule_embeddings(target_rule)
            rule_node_count = rule_graph.number_of_nodes()
            rule_edge_count = rule_graph.number_of_edges()
            print(f"[DEBUG] rule_node_count: {rule_node_count}, rule_edge_count: {rule_edge_count}")
            print(f"[DEBUG] rule_graph nodes: {list(rule_graph.nodes())}")
            print(f"[DEBUG] rule_graph edges: {list(rule_graph.edges(data=True))}")
            # 结构匹配先行 - 子图扩展
            if self.kg is None:
                raise ValueError("知识图谱未加载")
            candidate_subgraphs = self.graph_expander.expand_from_anchor(
                anchor, self.kg, rule_struct_embed, self,
                rule_node_count=rule_node_count,
                rule_edge_count=rule_edge_count
            )
            if not candidate_subgraphs:
                return MatchingResult(
                    rule_id=target_rule.get('rule_id', 'unknown'),
                    anchor=anchor,
                    matched_subgraphs=[],
                    structure_score=0.0,
                    semantic_score=0.0,
                    final_score=0.0,
                    is_matched=False,
                    processing_time=time.time() - start_time
                )
            # === 自动集成GraphMatcher+edge_match/node_match严格过滤 ===
            from networkx.algorithms.isomorphism import categorical_edge_match, categorical_node_match
            # 判断规则图节点是否有type属性
            has_node_type = any('type' in rule_graph.nodes[n] for n in rule_graph.nodes)
            edge_match = categorical_edge_match('relation', None)
            node_match = categorical_node_match('type', None) if has_node_type else None
            best_subgraph = None
            best_structure_score = 0.0
            best_mapping = None
            for subgraph in candidate_subgraphs:
                if node_match:
                    GM = isomorphism.GraphMatcher(subgraph, rule_graph, node_match=node_match, edge_match=edge_match)
                else:
                    GM = isomorphism.GraphMatcher(subgraph, rule_graph, edge_match=edge_match)
                if GM.is_isomorphic():
                    subgraph_struct_embed = self.get_structure_embedding(subgraph)
                    structure_score = self.struct_matcher.match_score(
                        subgraph_struct_embed, rule_struct_embed
                    )
                    if structure_score > best_structure_score:
                        best_structure_score = structure_score
                        best_subgraph = subgraph
                        best_mapping = dict(GM.mapping)  # 变量->实体
            if best_subgraph is None:
                return MatchingResult(
                    rule_id=target_rule.get('rule_id', 'unknown'),
                    anchor=anchor,
                    matched_subgraphs=[],
                    structure_score=0.0,
                    semantic_score=0.0,
                    final_score=0.0,
                    is_matched=False,
                    processing_time=time.time() - start_time
                )
            # 3. 增强语义验证（新增语义特征，不改变核心流程）
            best_subgraph_struct_embed = self.get_structure_embedding(best_subgraph)
            is_semantic_valid, semantic_score = self.semantic_verifier.verify_with_enhanced_semantics(
                best_subgraph, rule_semantic_embed, best_subgraph_struct_embed
            )
            # 4. 综合评分（保持原有逻辑，增加语义权重）
            final_score = self.scoring_function.compute_final_score(
                best_structure_score, semantic_score,
                structure_weight=self.config.structure_weight,  # 使用config参数
                semantic_weight=self.config.semantic_weight     # 使用config参数
            )
            # 更灵活的匹配条件：只要结构分数和最终分数达标即可
            # 语义验证作为辅助，不强制要求
            is_matched = (best_structure_score >= self.config.similarity_threshold and 
                         final_score >= self.config.final_threshold)
            # 如果语义验证通过，给予额外奖励
            if is_semantic_valid:
                final_score = min(1.0, final_score + 0.1)  # 语义验证通过给予0.1的奖励
            return MatchingResult(
                rule_id=target_rule.get('rule_id', 'unknown'),
                anchor=anchor,
                matched_subgraphs=[{
                    'subgraph': best_subgraph,
                    'structure_score': best_structure_score,
                    'semantic_score': semantic_score,
                    'variable_entity_mapping': best_mapping  # 新增映射输出
                }],
                structure_score=best_structure_score,
                semantic_score=semantic_score,
                final_score=final_score,
                is_matched=is_matched,
                processing_time=time.time() - start_time
            )
        except Exception as e:
            self.logger.error(f"子图匹配失败: {e}")
            return MatchingResult(
                rule_id=target_rule.get('rule_id', 'unknown'),
                anchor=anchor,
                matched_subgraphs=[],
                structure_score=0.0,
                semantic_score=0.0,
                final_score=0.0,
                is_matched=False,
                processing_time=time.time() - start_time
            )
    
    def run_batch_matching(self, rule_idx: int = 0) -> List[MatchingResult]:
        """批量匹配anchors到指定规则"""
        if not self.rules or rule_idx >= len(self.rules):
            raise ValueError(f"规则索引 {rule_idx} 超出范围")
        
        target_rule = self.rules[rule_idx]
        
        # 收集规则对应的anchor
        anchors = self.collect_anchors_for_rule(target_rule)
        
        if not anchors:
            self.logger.warning("没有收集到anchor")
            return []
        
        self.logger.info(f"开始批量匹配，共{len(anchors)}个anchor")
        
        results = []
        for i, anchor in enumerate(anchors):
            if i % 100 == 0:  # 每100个打印一次进度
                self.logger.info(f"处理第{i+1}/{len(anchors)}个anchor")
            
            result = self.match_subgraph_to_rule(anchor, target_rule)
            results.append(result)
            
            if result.is_matched:
                self.logger.info(f"匹配成功: anchor {anchor}, 得分: {result.final_score:.4f}")
        
        self.logger.info(f"批量匹配完成: {sum(1 for r in results if r.is_matched)}/{len(results)} 成功匹配")
        return results
    
    def evaluate_performance(self, results: List[MatchingResult], 
                           ground_truth: Optional[List[bool]] = None) -> EvaluationMetrics:
        """评估系统性能"""
        if not results:
            return EvaluationMetrics()
        
        # 计算平均处理时间
        avg_time = sum(r.processing_time for r in results) / len(results)
        
        # 如果有ground truth，计算准确率等指标
        if ground_truth and len(ground_truth) == len(results):
            correct_matches = sum(1 for r, gt in zip(results, ground_truth) 
                                if r.is_matched == gt)
            accuracy = correct_matches / len(results)
            
            # 计算精确率和召回率
            predicted_positives = sum(1 for r in results if r.is_matched)
            actual_positives = sum(ground_truth)
            
            precision = (sum(1 for r, gt in zip(results, ground_truth) 
                           if r.is_matched and gt) / predicted_positives 
                        if predicted_positives > 0 else 0.0)
            
            recall = (sum(1 for r, gt in zip(results, ground_truth) 
                         if r.is_matched and gt) / actual_positives 
                     if actual_positives > 0 else 0.0)
            
            return EvaluationMetrics(
                e2e_accuracy=accuracy,
                precision=precision,
                recall=recall,
                avg_processing_time=avg_time
            )
        else:
            # 没有ground truth时，只返回处理时间
            return EvaluationMetrics(avg_processing_time=avg_time)
    
    def save_results(self, results: List[MatchingResult], output_path: str = "results.json"):
        """保存匹配结果"""
        output_data = {
            "matching_results": [
                {
                    "rule_id": r.rule_id,
                    "anchor": r.anchor,
                    "is_matched": r.is_matched,
                    "structure_score": float(r.structure_score),
                    "semantic_score": float(r.semantic_score),
                    "final_score": float(r.final_score),
                    "processing_time": float(r.processing_time),
                    "matched_subgraphs": [
                        {
                            "nodes": list(subgraph["subgraph"].nodes()) if subgraph["subgraph"] else [],
                            "edges": [
                                (h, data.get('relation', None), t)
                                for h, t, data in subgraph["subgraph"].edges(data=True)
                            ] if subgraph["subgraph"] else [],
                            "structure_score": float(subgraph["structure_score"]),
                            "semantic_score": float(subgraph["semantic_score"]),
                            "variable_entity_mapping": subgraph.get("variable_entity_mapping", None)  # 新增映射输出
                        }
                        for subgraph in r.matched_subgraphs
                    ]
                }
                for r in results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存到: {output_path}")

def main():
    """主函数"""
    # 初始化系统
    config = Config()
    system = StructMatchSystem(config)
    
    # 加载真实数据 - 使用新的规则数据集
    print("=== 使用新的规则数据集 ===")
    
    # 检查是否有新的规则文件
    rules_path = "data/wn18rr_comprehensive_rules_structmatch.json"
    kg_path = "dataseet/wn18rr/train.txt"
    
    if not os.path.exists(rules_path):
        print(f"错误：未找到规则文件 {rules_path}")
        print("请确保规则文件存在")
        return
    
    if not os.path.exists(kg_path):
        print(f"错误：未找到知识图谱文件 {kg_path}")
        print("请确保知识图谱文件存在")
        return
    
    print(f"使用规则文件: {rules_path}")
    print(f"使用知识图谱: {kg_path}")
    
    # 加载数据
    system.load_data(kg_path, rules_path)
    
    print(f"\n总共加载了 {len(system.rules) if system.rules else 0} 条规则")
    
    print("\n开始结构匹配...")
    
    # 处理所有规则
    all_results = []
    total_matched = 0
    
    if not system.rules:
        print("错误：没有加载到规则")
        return
    
    for rule_idx in range(len(system.rules)):
        print(f"\n=== 处理规则 {rule_idx+1}/{len(system.rules)} ===")
        
        # 执行批量匹配
        results = system.run_batch_matching(rule_idx=rule_idx)
        all_results.extend(results)
        
        # 统计当前规则的匹配结果
        rule_matched = sum(1 for r in results if r.is_matched)
        total_matched += rule_matched
        print(f"规则 {rule_idx+1} 匹配成功: {rule_matched}/{len(results)} 个anchor")
    
    print(f"\n=== 所有规则处理完成 ===")
    print(f"总共处理: {len(all_results)} 个anchor")
    print(f"总共匹配成功: {total_matched} 个anchor")
    
    # 评估性能
    metrics = system.evaluate_performance(all_results)
    print(f"\n性能评估:")
    print(f"平均处理时间: {metrics.avg_processing_time:.4f}秒")
    if metrics.e2e_accuracy > 0:
        print(f"端到端准确率: {metrics.e2e_accuracy:.4f}")
        print(f"精确率: {metrics.precision:.4f}")
        print(f"召回率: {metrics.recall:.4f}")
    
    # 保存结果
    system.save_results(all_results)
    
    # 输出详细结果
    print(f"\n详细匹配结果:")
    matched_count = 0
    for i, result in enumerate(all_results):
        if result.is_matched:
            matched_count += 1
            if matched_count <= 5:  # 只显示前5个匹配结果
                print(f"\nAnchor {i+1}: {result.anchor}")
                print(f"规则ID: {result.rule_id}")
                print(f"匹配成功: {result.is_matched}")
                print(f"结构得分: {result.structure_score:.4f}")
                print(f"语义得分: {result.semantic_score:.4f}")
                print(f"最终得分: {result.final_score:.4f}")
    
    print(f"\n总共匹配成功: {matched_count} 个anchor")
    print(f"\n结果已保存到 results.json")
    print(f"可以运行 python evaluate_results.py 进行详细评估")

if __name__ == "__main__":
    main()