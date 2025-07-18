import itertools
import json
import subprocess
import re
import os
from utils.config import Config
from main import StructMatchSystem

# 参数网格
param_grid = {
    'beam_size': [3, 5, 7],
    'max_expand_steps': [3, 5],
    'similarity_threshold': [0.6, 0.7, 0.8],
    'final_threshold': [0.4, 0.5, 0.6],
}

# 数据路径（请根据实际情况修改）
kg_path = "dataseet/wn18rr/train.txt"
rules_path = "data/wn18rr_strict_structural_rules_fixed.json"

best_f1 = 0
best_params = None
results_log = []

def run_evaluation():
    """运行evaluate_results.py并解析输出"""
    try:
        result = subprocess.run(['python', 'evaluate_results.py'], 
                              capture_output=True, text=True)
        output = result.stdout
        
        # 解析输出中的F1值
        f1_match = re.search(r'F1: ([\d.]+)', output)
        precision_match = re.search(r'精确率: ([\d.]+)', output)
        recall_match = re.search(r'召回率: ([\d.]+)', output)
        
        f1 = float(f1_match.group(1)) if f1_match else 0.0
        precision = float(precision_match.group(1)) if precision_match else 0.0
        recall = float(recall_match.group(1)) if recall_match else 0.0
        
        return f1, precision, recall
    except Exception as e:
        print(f"评估失败: {e}")
        return 0.0, 0.0, 0.0

for values in itertools.product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), values))
    print(f"\n=== 测试参数: {params} ===")
    
    # 运行推理
    config = Config(**params)
    system = StructMatchSystem(config)
    system.load_data(kg_path, rules_path)
    results = system.run_batch_matching(rule_idx=0)
    system.save_results(results)  # 保存到results.json
    
    # 运行评估
    f1, precision, recall = run_evaluation()
    print(f"精确率: {precision:.4f} 召回率: {recall:.4f} F1: {f1:.4f}")
    
    results_log.append({
        'params': params,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })
    
    if f1 > best_f1:
        best_f1 = f1
        best_params = params

print("\n=== 最优参数 ===")
print(f"最优参数: {best_params}")
print(f"最优F1: {best_f1:.4f}")

# 保存所有结果
with open("grid_search_results.json", "w", encoding="utf-8") as f:
    json.dump(results_log, f, indent=2, ensure_ascii=False) 