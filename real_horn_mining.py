import os
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import itertools

# 配置
DATA_DIR = 'dataseet/wn18rr/'
OUTPUT_DIR = 'data/'
TRIPLE_FILES = ['train.txt']
RULE_MIN_SUPPORT = 5
RULE_MIN_CONFIDENCE = 0.6

# 定义规则结构模板
RULE_TEMPLATES = {
    'chain': {
        'premise': [('X', 'r1', 'Y'), ('Y', 'r2', 'Z')],
        'conclusion': ('X', 'r3', 'Z'),
        'description': '链式结构: (X,r1,Y)∧(Y,r2,Z)⇒(X,r3,Z)'
    },
    'y_type': {
        'premise': [('X', 'r1', 'Y'), ('X', 'r2', 'Z')],
        'conclusion': ('Y', 'r3', 'Z'),
        'description': 'Y型结构: (X,r1,Y)∧(X,r2,Z)⇒(Y,r3,Z)'
    },
    'v_type': {
        'premise': [('X', 'r1', 'Y'), ('Z', 'r2', 'Y')],
        'conclusion': ('X', 'r3', 'Z'),
        'description': 'V型结构: (X,r1,Y)∧(Z,r2,Y)⇒(X,r3,Z)'
    }
}

print("=== WN18RR 严格结构规则挖掘（重构版） ===")
print(f"配置: 支持度≥{RULE_MIN_SUPPORT}, 置信度≥{RULE_MIN_CONFIDENCE}")

class RuleMiner:
    def __init__(self, triples, min_support=5, min_confidence=0.6):
        self.triples = triples
        self.triples_set = set(triples)
        self.min_support = min_support
        self.min_confidence = min_confidence
        
        # 构建索引
        self.r2ht = defaultdict(list)  # relation -> [(head, tail)]
        self.h2rt = defaultdict(list)  # head -> [(relation, tail)]
        self.t2rh = defaultdict(list)  # tail -> [(relation, head)]
        self.entities = set()
        self.relations = set()
        
        self._build_indexes()
    
    def _build_indexes(self):
        """构建高效的索引结构"""
        print("构建索引...")
        for h, r, t in self.triples:
            self.r2ht[r].append((h, t))
            self.h2rt[h].append((r, t))
            self.t2rh[t].append((r, h))
            self.entities.add(h)
            self.entities.add(t)
            self.relations.add(r)
        
        print(f"实体数: {len(self.entities)}, 关系数: {len(self.relations)}")
    
    def find_rule_instances(self, template_name, r1, r2, r3):
        """根据模板和关系组合查找规则实例"""
        template = RULE_TEMPLATES[template_name]
        instances = []
        
        if template_name == 'chain':
            # 链式: (X,r1,Y)∧(Y,r2,Z)⇒(X,r3,Z)
            for h1, t1 in self.r2ht[r1]:
                for h2, t2 in self.r2ht[r2]:
                    if t1 == h2:  # Y连接
                        x, y, z = h1, t1, t2
                        if (x, r1, y) in self.triples_set and (y, r2, z) in self.triples_set:
                            conclusion = (x, r3, z)
                            instances.append({
                                'premise_instance': [(x, r1, y), (y, r2, z)],
                                'conclusion_instance': conclusion,
                                'conclusion_in_kg': conclusion in self.triples_set
                            })
        
        elif template_name == 'y_type':
            # Y型: (X,r1,Y)∧(X,r2,Z)⇒(Y,r3,Z)
            for h1, t1 in self.r2ht[r1]:
                for h2, t2 in self.r2ht[r2]:
                    if h1 == h2 and t1 != t2:  # 同一个头实体
                        x, y, z = h1, t1, t2
                        if (x, r1, y) in self.triples_set and (x, r2, z) in self.triples_set:
                            conclusion = (y, r3, z)
                            instances.append({
                                'premise_instance': [(x, r1, y), (x, r2, z)],
                                'conclusion_instance': conclusion,
                                'conclusion_in_kg': conclusion in self.triples_set
                            })
        
        elif template_name == 'v_type':
            # V型: (X,r1,Y)∧(Z,r2,Y)⇒(X,r3,Z)
            for h1, t1 in self.r2ht[r1]:
                for h2, t2 in self.r2ht[r2]:
                    if t1 == t2 and h1 != h2:  # 同一个尾实体
                        x, y, z = h1, t1, h2
                        if (x, r1, y) in self.triples_set and (z, r2, y) in self.triples_set:
                            conclusion = (x, r3, z)
                            instances.append({
                                'premise_instance': [(x, r1, y), (z, r2, y)],
                                'conclusion_instance': conclusion,
                                'conclusion_in_kg': conclusion in self.triples_set
                            })
        
        return instances
    
    def mine_rules(self, max_relations=None):
        """挖掘所有规则"""
        print("开始挖掘规则...")
        
        relations = list(self.relations)
        if max_relations:
            relations = relations[:max_relations]
        
        print(f"测试关系数: {len(relations)}")
        
        all_rules = []
        
        # 为每种模板挖掘规则
        for template_name, template in RULE_TEMPLATES.items():
            print(f"\n挖掘 {template['description']}...")
            
            # 生成所有可能的关系组合
            relation_combinations = list(itertools.product(relations, repeat=3))
            
            for r1, r2, r3 in tqdm(relation_combinations, desc=f"{template_name}"):
                instances = self.find_rule_instances(template_name, r1, r2, r3)
                
                if len(instances) >= self.min_support:
                    support = len(instances)
                    conclusion_matches = sum(1 for inst in instances if inst['conclusion_in_kg'])
                    confidence = conclusion_matches / support if support > 0 else 0
                    
                    if confidence >= self.min_confidence:
                        rule_data = {
                            'template': template_name,
                            'premise': [(template['premise'][0][0], r1, template['premise'][0][2]),
                                       (template['premise'][1][0], r2, template['premise'][1][2])],
                            'conclusion': (template['conclusion'][0], r3, template['conclusion'][2]),
                            'support': support,
                            'confidence': confidence,
                            'instances': instances
                        }
                        all_rules.append(rule_data)
        
        return all_rules
    
    def mine_rules_optimized(self, max_relations=None):
        """优化的规则挖掘：只考虑有足够支持的关系组合"""
        print("开始优化挖掘规则...")
        
        relations = list(self.relations)
        if max_relations:
            relations = relations[:max_relations]
        
        print(f"测试关系数: {len(relations)}")
        
        # 预筛选有足够支持的关系
        relation_support = defaultdict(int)
        for r in relations:
            relation_support[r] = len(self.r2ht[r])
        
        # 只考虑支持度较高的关系
        min_relation_support = 10
        candidate_relations = [r for r in relations if relation_support[r] >= min_relation_support]
        print(f"候选关系数: {len(candidate_relations)}")
        
        all_rules = []
        
        for template_name, template in RULE_TEMPLATES.items():
            print(f"\n挖掘 {template['description']}...")
            
            relation_combinations = list(itertools.product(candidate_relations, repeat=3))
            
            for r1, r2, r3 in tqdm(relation_combinations, desc=f"{template_name}"):
                instances = self.find_rule_instances(template_name, r1, r2, r3)
                
                if len(instances) >= self.min_support:
                    support = len(instances)
                    conclusion_matches = sum(1 for inst in instances if inst['conclusion_in_kg'])
                    confidence = conclusion_matches / support if support > 0 else 0
                    
                    if confidence >= self.min_confidence:
                        rule_data = {
                            'template': template_name,
                            'premise': [(template['premise'][0][0], r1, template['premise'][0][2]),
                                       (template['premise'][1][0], r2, template['premise'][1][2])],
                            'conclusion': (template['conclusion'][0], r3, template['conclusion'][2]),
                            'support': support,
                            'confidence': confidence,
                            'instances': instances
                        }
                        all_rules.append(rule_data)
        
        return all_rules

def main():
    # 加载三元组
    print("1. 加载三元组...")
    triples = []
    for fname in TRIPLE_FILES:
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        triples.append(tuple(parts))
    
    print(f"三元组总数: {len(triples)}")
    
    # 创建挖掘器
    miner = RuleMiner(triples, RULE_MIN_SUPPORT, RULE_MIN_CONFIDENCE)
    
    # 挖掘规则
    print("2. 挖掘规则...")
    rules = miner.mine_rules_optimized(max_relations=20)  # 限制关系数以提高效率
    
    # 输出结果
    print(f"\n总共挖掘到 {len(rules)} 条高质量规则")
    
    for rule in rules:
        print(f"规则: {rule['premise'][0]} ∧ {rule['premise'][1]} ⇒ {rule['conclusion']}")
        print(f"  模板: {rule['template']}, 支持度: {rule['support']}, 置信度: {rule['confidence']:.3f}")
    
    # 保存规则
    if rules:
        output_file = os.path.join(OUTPUT_DIR, 'wn18rr_strict_structural_rules_v2.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
        print(f"\n规则已保存到 {output_file}")
    else:
        print("没有挖掘到满足条件的规则")
    
    print("挖掘完成!")

if __name__ == "__main__":
    main() 