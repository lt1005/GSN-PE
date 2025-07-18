import json
import os
from collections import defaultdict

def canonical_anchor(anchor):
    """将anchor归一化为标准形式，支持路径正反等价"""
    # anchor是一个三元组，比如 ("A", "relation", "B")
    # 对于路径结构，正反都算等价
    # 这里简单处理：取字典序较小的作为标准形式
    if isinstance(anchor, (list, tuple)) and len(anchor) == 3:
        # 如果是路径结构（两个实体不同），可以考虑正反等价
        if anchor[0] != anchor[2]:  # 头尾实体不同
            # 创建反向anchor
            reverse_anchor = (anchor[2], anchor[1], anchor[0])
            # 取字典序较小的
            return min(anchor, reverse_anchor)
    return anchor

def main():
    """主函数"""
    # 检查结果文件
    results_file = "results.json"  # 修改为result2.json
    if not os.path.exists(results_file):
        print(f"错误：结果文件 {results_file} 不存在")
        print("请先运行 main.py 生成结果文件")
        return
    
    print(f"=== 读取结果文件: {results_file} ===")
    
    # 读取结果
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    print(f"读取到 {len(results_data['matching_results'])} 条匹配结果")
    
    # 评估结果
    print("\n=== 每条规则单独评估（支持anchor归一化）===")
    
    # 加载新的规则文件
    rules_path = 'data/wn18rr_comprehensive_rules_structmatch.json'
    instances_path = 'data/wn18rr_comprehensive_rules.json'
    
    # 检查文件是否存在
    if not os.path.exists(rules_path):
        print(f"错误：规则文件不存在 {rules_path}")
        return
    
    if not os.path.exists(instances_path):
        print(f"错误：实例文件不存在 {instances_path}")
        return
    
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    
    with open(instances_path, 'r', encoding='utf-8') as f:
        instances_data = json.load(f)
    
    # 按规则分组
    rule_results = defaultdict(list)
    for result in results_data['matching_results']:
        rule_id = result['rule_id']
        rule_results[rule_id].append(result)
    
    # 评估每条规则
    for rule_id, rule_result_list in rule_results.items():
        evaluate_rule_performance(rule_id, rule_result_list, instances_data)
    
    print("\n=== 评估完成 ===")

def evaluate_rule_performance(rule_id, rule_result_list, instances_data):
    """评估单条规则的性能"""
    # 找到对应的规则实例
    rule_instances = None
    for rule_data in instances_data:
        if rule_data.get('rule_id') == rule_id:
            rule_instances = rule_data.get('instances', [])
            break
    
    if rule_instances is None:
        print(f"警告：未找到规则 {rule_id} 的实例")
        return
    
    # 真实实例anchor集合（不使用归一化，保持与推理一致）
    gt_anchors = set(tuple(inst['premise_instance'][0]) for inst in rule_instances)
    
    # 推理发现的anchor集合（不使用归一化，保持与推理一致）
    pred_anchors = set(tuple(r['anchor']) for r in rule_result_list if r['is_matched'])
    
    # 评估
    tp = len(pred_anchors & gt_anchors)
    fp = len(pred_anchors - gt_anchors)
    fn = len(gt_anchors - pred_anchors)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    
    print(f"规则: {rule_id}")
    print(f"  实例数: {len(rule_instances)}")
    print(f"  唯一anchor数: {len(gt_anchors)}")
    print(f"  推理发现: {len(pred_anchors)}")
    print(f"  正确匹配: {tp}")
    print(f"  精确率: {precision:.4f}  召回率: {recall:.4f}  F1: {f1:.4f}")
    
    # 输出详细对比结果
    # print('  推理发现但不在真实实例中的anchor:')
    # for anchor in sorted(pred_anchors - gt_anchors):
    #     print('    ', anchor)
    # print('  真实实例但未被发现的anchor:')
    # for anchor in sorted(gt_anchors - pred_anchors):
    #     print('    ', anchor)
    # print()

if __name__ == "__main__":
    main() 