#!/usr/bin/env python3
"""
关系相似度匹配工具
解决关系重叠少的问题，提高链式结构召回率
"""

import json
import networkx as nx
from collections import defaultdict
import numpy as np

def load_data():
    """加载数据"""
    print("=== 加载数据 ===")
    
    # 加载知识图谱
    kg = nx.Graph()
    with open("dataseet/wn18rr/train.txt", "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            kg.add_edge(h, t, relation=r)
    print(f"知识图谱: {kg.number_of_nodes()} 实体, {kg.number_of_edges()} 边")
    
    # 加载规则
    with open("data/wn18rr_strict_structural_rules_fixed.json", "r", encoding="utf-8") as f:
        rules = json.load(f)
    print(f"规则数: {len(rules)}")
    
    return kg, rules

def calculate_relation_similarity(kg):
    """计算关系相似度"""
    print("\n=== 计算关系相似度 ===")
    
    # 统计关系分布
    relation_stats = defaultdict(lambda: {'count': 0, 'entities': set(), 'patterns': defaultdict(int)})
    
    for h, t in kg.edges():
        edge_data = kg.get_edge_data(h, t)
        if edge_data:
            rel = edge_data.get('relation', 'unknown')
            relation_stats[rel]['count'] += 1
            relation_stats[rel]['entities'].add(h)
            relation_stats[rel]['entities'].add(t)
            
            # 统计实体度分布
            h_degree = len(list(kg.neighbors(h)))
            t_degree = len(list(kg.neighbors(t)))
            relation_stats[rel]['patterns'][(h_degree, t_degree)] += 1
    
    # 计算关系相似度矩阵
    relations = list(relation_stats.keys())
    similarity_matrix = {}
    
    for i, rel1 in enumerate(relations):
        similarity_matrix[rel1] = {}
        for j, rel2 in enumerate(relations):
            if i == j:
                similarity_matrix[rel1][rel2] = 1.0
            else:
                # 计算Jaccard相似度
                entities1 = relation_stats[rel1]['entities']
                entities2 = relation_stats[rel2]['entities']
                
                intersection = len(entities1.intersection(entities2))
                union = len(entities1.union(entities2))
                
                jaccard = intersection / union if union > 0 else 0.0
                
                # 计算模式相似度
                patterns1 = relation_stats[rel1]['patterns']
                patterns2 = relation_stats[rel2]['patterns']
                
                pattern_similarity = 0.0
                if patterns1 and patterns2:
                    common_patterns = set(patterns1.keys()).intersection(set(patterns2.keys()))
                    if common_patterns:
                        pattern_similarity = len(common_patterns) / len(set(patterns1.keys()).union(set(patterns2.keys())))
                
                # 综合相似度
                similarity = 0.7 * jaccard + 0.3 * pattern_similarity
                similarity_matrix[rel1][rel2] = similarity
    
    return similarity_matrix, relation_stats

def find_similar_relations(target_rel, similarity_matrix, threshold=0.3):
    """找到相似关系"""
    if target_rel not in similarity_matrix:
        return []
    
    similar_relations = []
    for rel, similarity in similarity_matrix[target_rel].items():
        if similarity >= threshold and rel != target_rel:
            similar_relations.append((rel, similarity))
    
    # 按相似度排序
    similar_relations.sort(key=lambda x: x[1], reverse=True)
    return similar_relations

def test_relation_similarity_matching(kg, rules, similarity_matrix):
    """测试关系相似度匹配"""
    print("\n=== 测试关系相似度匹配 ===")
    
    # 测试rule_001的链式结构
    rule_001 = rules[0]
    print(f"测试规则: {rule_001['rule_id']}")
    print(f"规则前提: {rule_001['premise']}")
    
    # 分析需要的扩展
    premise = rule_001['premise']
    step1_rel = premise[0][1]  # _also_see
    step2_rel = premise[1][1]  # _synset_domain_topic_of
    
    print(f"\n第一步关系: {step1_rel}")
    print(f"第二步关系: {step2_rel}")
    
    # 找到相似关系
    similar_step2_rels = find_similar_relations(step2_rel, similarity_matrix, threshold=0.2)
    print(f"\n与 {step2_rel} 相似的关系:")
    for rel, sim in similar_step2_rels[:5]:
        print(f"  {rel}: {sim:.3f}")
    
    # 测试一个具体的锚点
    test_anchor = ['01976089', '_also_see', '01974062']
    h, r, t = test_anchor
    
    print(f"\n测试锚点: {test_anchor}")
    
    # 检查精确匹配
    neighbors = list(kg.neighbors(t))
    exact_matches = []
    for neighbor in neighbors:
        edge_data = kg.get_edge_data(t, neighbor)
        if edge_data and edge_data.get('relation') == step2_rel:
            exact_matches.append((t, step2_rel, neighbor))
    
    print(f"精确匹配数量: {len(exact_matches)}")
    
    # 检查相似关系匹配
    similar_matches = []
    for rel, sim in similar_step2_rels:
        for neighbor in neighbors:
            edge_data = kg.get_edge_data(t, neighbor)
            if edge_data and edge_data.get('relation') == rel:
                similar_matches.append((t, rel, neighbor, sim))
    
    print(f"相似关系匹配数量: {len(similar_matches)}")
    
    if similar_matches:
        print(f"相似关系匹配示例:")
        for i, (h_rel, r_rel, t_rel, sim) in enumerate(similar_matches[:3]):
            print(f"  {i+1}: ({h_rel}, {r_rel}, {t_rel}) - 相似度: {sim:.3f}")
            
            # 检查结论
            conclusion = [h, step2_rel, t_rel]
            if kg.has_edge(h, t_rel):
                edge_data = kg.get_edge_data(h, t_rel)
                kg_rel = edge_data.get('relation') if edge_data else None
                print(f"    结论: {conclusion} - 在KG中: 是, 关系: {kg_rel}")
            else:
                print(f"    结论: {conclusion} - 在KG中: 否")

def create_enhanced_matching_config():
    """创建增强匹配配置"""
    print("\n=== 创建增强匹配配置 ===")
    
    enhanced_config = {
        "similarity_threshold": 0.01,
        "semantic_threshold": 0.01,
        "final_threshold": 0.01,
        "isomorphic_threshold": 0.05,
        "structure_weight": 0.8,
        "semantic_weight": 0.2,
        "max_expand_steps": 20,
        "beam_size": 50,
        "motif_threshold": 0.5,
        "struct_score_delta": 0.0,
        "max_candidates": 500,
        "use_trained_model": False,
        "rules_file": "data/wn18rr_strict_structural_rules_fixed.json",
        "enable_relation_similarity": True,
        "relation_similarity_threshold": 0.2,
        "enable_flexible_matching": True,
        "flexible_matching_threshold": 0.1,
        "max_similar_relations": 5
    }
    
    with open("enhanced_config.json", "w", encoding="utf-8") as f:
        json.dump(enhanced_config, f, indent=2, ensure_ascii=False)
    
    print("✅ 创建了增强配置: enhanced_config.json")
    print("主要改进:")
    print("1. 启用关系相似度匹配")
    print("2. 关系相似度阈值: 0.2")
    print("3. 启用灵活匹配")
    print("4. 增加扩展能力")

def main():
    """主函数"""
    print("关系相似度匹配工具")
    print("=" * 60)
    
    try:
        # 加载数据
        kg, rules = load_data()
        
        # 计算关系相似度
        similarity_matrix, relation_stats = calculate_relation_similarity(kg)
        
        # 测试关系相似度匹配
        test_relation_similarity_matching(kg, rules, similarity_matrix)
        
        # 创建增强配置
        create_enhanced_matching_config()
        
        print("\n=== 总结 ===")
        print("解决方案:")
        print("1. ✅ 已更新 fixed_config.json 降低阈值")
        print("2. ✅ 已创建 enhanced_config.json 启用关系相似度")
        print("3. 下一步: 使用新配置重新运行推理")
        
        print("\n使用步骤:")
        print("1. 复制 enhanced_config.json 到 fixed_config.json")
        print("2. 运行 python main.py")
        print("3. 运行 python evaluation.py")
        print("4. 检查召回率是否提升")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 