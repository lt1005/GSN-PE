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
from matcher.semantic_verifier import SemanticVerifier
from trainer.train_gnn import GNNTrainer
from trainer.contrastive_loss import ContrastiveLoss

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
    """StructMatch主系统 - 基于SRS重构"""
    
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
        self.scoring_function = ScoringFunction()
        self.semantic_verifier = SemanticVerifier(
            entity_encoder=None,  # 需要根据实际语义模型初始化
            relation_encoder=None,
            threshold=config.semantic_threshold
        )
        
        # 知识图谱和规则
        self.kg = None
        self.rules = None
        
        # 评估指标
        self.evaluation_metrics = EvaluationMetrics()
        
        self.logger.info("StructMatch系统初始化完成")
    
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
    
    def get_structure_embedding(self, graph: nx.Graph) -> torch.Tensor:
        """获取图的结构嵌入"""
        try:
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
        except Exception as e:
            self.logger.error(f"结构嵌入提取失败: {e}")
            raise
    
    def get_semantic_embedding(self, graph: nx.Graph) -> torch.Tensor:
        """获取图的语义嵌入"""
        try:
            return self.semantic_verifier.encode_semantic(graph)
        except Exception as e:
            self.logger.error(f"语义嵌入提取失败: {e}")
            # 返回零向量，避免语义验证误接受
            return torch.zeros(self.config.semantic_dim)
    
    def precompute_rule_embeddings(self, rule: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """预计算规则的结构嵌入和语义嵌入"""
        try:
            # 构建规则前件图
            rule_graph = nx.Graph()
            for triple in rule['premise']:
                h, r, t = triple['h'], triple['r'], triple['t']
                rule_graph.add_edge(h, t, relation=r)
            
            # 计算结构嵌入
            struct_embed = self.get_structure_embedding(rule_graph)
            
            # 计算语义嵌入
            semantic_embed = self.get_semantic_embedding(rule_graph)
            
            return struct_embed, semantic_embed
        except Exception as e:
            self.logger.error(f"规则嵌入预计算失败: {e}")
            raise
    
    def match_subgraph_to_rule(self, anchor: Tuple[str, str, str], 
                              target_rule: Dict) -> MatchingResult:
        """匹配单个anchor到规则"""
        start_time = time.time()
        
        try:
            # 预计算规则嵌入
            rule_struct_embed, rule_semantic_embed = self.precompute_rule_embeddings(target_rule)
            
            # 1. 结构匹配先行 - 子图扩展
            if self.kg is None:
                raise ValueError("知识图谱未加载")
            
            candidate_subgraphs = self.graph_expander.expand_from_anchor(
                anchor, self.kg, rule_struct_embed, self
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
            
            # 2. 结构匹配筛选
            struct_matched_subgraphs = []
            for subgraph in candidate_subgraphs:
                struct_score = self.scoring_function.similarity_score(
                    self.get_structure_embedding(subgraph), 
                    rule_struct_embed
                )
                
                if struct_score >= self.config.similarity_threshold:
                    struct_matched_subgraphs.append((subgraph, struct_score))
            
            if not struct_matched_subgraphs:
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
            
            # 3. 语义验证紧随其后
            final_matched_subgraphs = []
            best_semantic_score = 0.0
            
            for subgraph, struct_score in struct_matched_subgraphs:
                sem_pass, sem_score = self.semantic_verifier.semantic_verify(
                    subgraph, rule_semantic_embed
                )
                
                if sem_pass:
                    final_score = 0.7 * struct_score + 0.3 * sem_score  # 加权融合
                    final_matched_subgraphs.append({
                        'nodes': list(subgraph.nodes()),
                        'edges': [(u, v, subgraph[u][v].get('relation', 'unknown')) 
                                 for u, v in subgraph.edges()],
                        'structure_score': struct_score,
                        'semantic_score': sem_score,
                        'final_score': final_score
                    })
                    best_semantic_score = max(best_semantic_score, sem_score)
            
            # 返回结果
            is_matched = len(final_matched_subgraphs) > 0
            avg_struct_score = sum(s[1] for s in struct_matched_subgraphs) / len(struct_matched_subgraphs) if struct_matched_subgraphs else 0.0
            
            return MatchingResult(
                rule_id=target_rule.get('rule_id', 'unknown'),
                anchor=anchor,
                matched_subgraphs=final_matched_subgraphs,
                structure_score=avg_struct_score,
                semantic_score=best_semantic_score,
                final_score=0.7 * avg_struct_score + 0.3 * best_semantic_score if is_matched else 0.0,
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
    
    def run_batch_matching(self, anchors: List[Tuple[str, str, str]], 
                          rule_idx: int = 0) -> List[MatchingResult]:
        """批量匹配anchors到指定规则"""
        if not self.rules or rule_idx >= len(self.rules):
            raise ValueError(f"规则索引 {rule_idx} 超出范围")
        
        target_rule = self.rules[rule_idx]
        self.logger.info(f"开始批量匹配，共{len(anchors)}个anchor")
        
        results = []
        for i, anchor in enumerate(anchors):
            self.logger.info(f"处理第{i+1}/{len(anchors)}个anchor")
            self.logger.info(f"开始处理anchor: {anchor}")
            
            result = self.match_subgraph_to_rule(anchor, target_rule)
            results.append(result)
            
            if result.is_matched:
                self.logger.info(f"匹配成功: {result.is_matched}")
                self.logger.info(f"匹配得分: {result.final_score:.4f}")
                if result.matched_subgraphs:
                    subgraph = result.matched_subgraphs[0]
                    self.logger.info(f"扩展子图节点: {subgraph['nodes']}")
                    self.logger.info(f"扩展子图边: {subgraph['edges']}")
            else:
                self.logger.info(f"匹配失败，处理时间: {result.processing_time:.4f}秒")
        
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
                            "nodes": subgraph["nodes"],
                            "edges": subgraph["edges"],
                            "structure_score": float(subgraph["structure_score"]),
                            "semantic_score": float(subgraph["semantic_score"]),
                            "final_score": float(subgraph["final_score"])
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

def create_sample_data():
    """创建示例数据文件"""
    # 创建示例知识图谱
    kg_data = [
        "A\tfriendOf\tB",
        "B\tworksWith\tC", 
        "D\tknows\tC",
        "A\tlivesIn\tTokyo",
        "B\tlivesIn\tOsaka",
        "C\tlivesIn\tTokyo",
        "D\tlivesIn\tOsaka",
        "E\tfriendOf\tF",
        "F\tworksWith\tG",
        "G\tknows\tH",
        "A\tknows\tD",
        "B\tknows\tE"
    ]
    
    with open("data/example_kg.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(kg_data))
    
    # 创建示例规则
    rules_data = [
        {
            "rule_id": "rule_001",
            "premise": [
                {"h": "A", "r": "friendOf", "t": "B"},
                {"h": "B", "r": "worksWith", "t": "C"},
                {"h": "D", "r": "knows", "t": "C"}
            ],
            "conclusion": {"h": "A", "r": "relatedTo", "t": "D"}
        },
        {
            "rule_id": "rule_002", 
            "premise": [
                {"h": "X", "r": "friendOf", "t": "Y"},
                {"h": "Y", "r": "livesIn", "t": "Z"}
            ],
            "conclusion": {"h": "X", "r": "knows", "t": "Z"}
        },
        {
            "rule_id": "rule_003",
            "premise": [
                {"h": "P", "r": "worksWith", "t": "Q"},
                {"h": "Q", "r": "knows", "t": "R"}
            ],
            "conclusion": {"h": "P", "r": "relatedTo", "t": "R"}
        }
    ]
    
    with open("data/rules.json", "w", encoding="utf-8") as f:
        json.dump(rules_data, f, indent=2, ensure_ascii=False)
    
    print("示例数据文件已创建:")
    print("- data/example_kg.txt")
    print("- data/rules.json")

def run_batch_rule_inference(system, rules_path, instances_path):
    """批量规则推理与评估"""
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    with open(instances_path, 'r', encoding='utf-8') as f:
        instances = json.load(f)
    # 按rule_id分组实例
    rule2instances = defaultdict(list)
    for inst in instances:
        rule2instances[inst['rule_id']].append(inst)
    all_results = []
    all_gt = []
    for rule in rules:
        rule_id = rule['rule_id']
        premise = rule['premise']
        conclusion = rule['conclusion']
        rule_instances = rule2instances.get(rule_id, [])
        # 构建规则图
        rule_graph = nx.Graph()
        for triple in premise:
            rule_graph.add_edge(triple['h'], triple['t'], relation=triple['r'])
        # 预计算结构/语义嵌入
        struct_embed = system.get_structure_embedding(rule_graph)
        semantic_embed = system.get_semantic_embedding(rule_graph)
        for inst in rule_instances:
            # 构建实例前提子图
            subgraph = nx.Graph()
            for h, r, t in inst['premise_instance']:
                subgraph.add_edge(h, t, relation=r)
            # 结构匹配
            struct_score = system.scoring_function.similarity_score(
                system.get_structure_embedding(subgraph), struct_embed)
            struct_pass = struct_score >= system.config.similarity_threshold
            # 语义验证
            sem_pass, sem_score = system.semantic_verifier.semantic_verify(subgraph, semantic_embed)
            final_score = 0.7 * struct_score + 0.3 * sem_score
            is_matched = struct_pass and sem_pass
            # ground truth
            gt = inst['conclusion_in_kg']
            all_results.append(is_matched)
            all_gt.append(gt)
    # 评估
    correct = sum(1 for p, g in zip(all_results, all_gt) if p and g)
    predicted = sum(all_results)
    actual = sum(all_gt)
    precision = correct / predicted if predicted else 0.0
    recall = correct / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"\n批量规则推理评估:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Total instances: {len(all_results)}")
    print(f"Correct matches: {correct}")
    print(f"Predicted matches: {predicted}")
    print(f"Ground truth positives: {actual}")

def main():
    """主函数"""
    # 创建示例数据
    create_sample_data()
    
    # 初始化系统
    config = Config()
    system = StructMatchSystem(config)
    
    # 加载数据
    system.load_data("data/example_kg.txt", "data/rules.json")
    
    print("\n开始结构匹配...")
    
    # 定义测试anchors
    test_anchors = [
        ('A', 'friendOf', 'B'),
        ('B', 'worksWith', 'D'),
        ('E', 'friendOf', 'F'),
        ('A', 'livesIn', 'Tokyo')
    ]
    
    # 执行批量匹配
    results = system.run_batch_matching(test_anchors, rule_idx=0)
    
    # 评估性能
    metrics = system.evaluate_performance(results)
    print(f"\n性能评估:")
    print(f"平均处理时间: {metrics.avg_processing_time:.4f}秒")
    if metrics.e2e_accuracy > 0:
        print(f"端到端准确率: {metrics.e2e_accuracy:.4f}")
        print(f"精确率: {metrics.precision:.4f}")
        print(f"召回率: {metrics.recall:.4f}")
    
    # 保存结果
    system.save_results(results)
    
    # 输出详细结果
    print(f"\n详细匹配结果:")
    for i, result in enumerate(results):
        if result.is_matched:
            print(f"\nAnchor {i+1}: {result.anchor}")
            print(f"匹配成功: {result.is_matched}")
            print(f"结构得分: {result.structure_score:.4f}")
            print(f"语义得分: {result.semantic_score:.4f}")
            print(f"最终得分: {result.final_score:.4f}")
            if result.matched_subgraphs:
                subgraph = result.matched_subgraphs[0]
                print(f"匹配子图节点: {subgraph['nodes']}")
                print(f"匹配子图边: {subgraph['edges']}")
        else:
            if 'anchor' in result.__dict__:
                print(f"\nAnchor {i+1}: {result.anchor} - 处理失败")
            else:
                print(f"\nAnchor {i+1}: 未知anchor - 处理失败，result内容: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rules', type=str, default=None, help='规则文件路径')
    parser.add_argument('--instances', type=str, default=None, help='实例ground truth文件路径')
    args = parser.parse_args()
    
    if args.rules and args.instances:
        # 批量规则推理模式
        config = Config()
        system = StructMatchSystem(config)
        system.load_data("data/example_kg.txt", "data/rules.json")  # 这里KG路径可根据需要调整
        run_batch_rule_inference(system, args.rules, args.instances)
    else:
        # 原有示例流程
        main()
