import os
import json
import random
from collections import defaultdict, Counter
from itertools import combinations, product

DATA_DIR = 'dataseet/wn18rr/'
OUTPUT_DIR = 'data/'
TRIPLE_FILES = ['train.txt', 'valid.txt', 'test.txt']
RULE_MIN_SUPPORT = 5
RULE_MIN_CONFIDENCE = 0.7
PREMISE_SIZE_RANGE = range(2, 6)  # 2~5
SPLIT_RATIO = [0.8, 0.1, 0.1]  # train/valid/test

# 1. 读取所有三元组
def load_triples():
    triples = []
    for fname in TRIPLE_FILES:
        path = os.path.join(DATA_DIR, fname)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    triples.append(tuple(parts))
    return triples

# 2. 构建索引
def build_index(triples):
    h2rt = defaultdict(list)
    t2hr = defaultdict(list)
    r2ht = defaultdict(list)
    for h, r, t in triples:
        h2rt[h].append((r, t))
        t2hr[t].append((h, r))
        r2ht[r].append((h, t))
    return h2rt, t2hr, r2ht

# 3. 规则挖掘（2~5条前提，链式/非链式都可）
def mine_rules(triples):
    # 统计所有实体、关系
    entities = set()
    relations = set()
    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)
    triples_set = set(triples)
    # 统计所有可能的前提组合
    premise_counter = Counter()
    rule_counter = Counter()
    # 只考虑变量化的规则（即用变量X/Y/Z等替换实体）
    # 这里只做简单枚举：同一组头实体的所有出边组合
    for size in PREMISE_SIZE_RANGE:
        for h in entities:
            out_edges = [(h, r, t) for (h1, r, t) in triples if h1 == h]
            if len(out_edges) < size:
                continue
            for premise in combinations(out_edges, size):
                # 结论三元组：任选一个与前提不同的三元组，头实体相同
                premise_set = set(premise)
                possible_conclusions = [tr for tr in out_edges if tr not in premise_set]
                for concl in possible_conclusions:
                    # 变量化：用X替换h，Y1/Y2...替换t
                    var_map = {h: 'X'}
                    t_vars = {}
                    for idx, (h0, r0, t0) in enumerate(premise):
                        if t0 not in var_map:
                            var_map[t0] = f'Y{idx+1}'
                    concl_h, concl_r, concl_t = concl
                    if concl_t not in var_map:
                        var_map[concl_t] = f'Yc'
                    # 变量化前提
                    premise_var = tuple((var_map[h0], r0, var_map[t0]) for (h0, r0, t0) in premise)
                    conclusion_var = (var_map[concl_h], concl_r, var_map[concl_t])
                    premise_counter[(premise_var, conclusion_var)] += 1
    # 统计规则实例
    rule_instances = defaultdict(list)
    for (premise_var, conclusion_var), count in premise_counter.items():
        if count < RULE_MIN_SUPPORT:
            continue
        # 统计置信度
        premise_matches = 0
        conclusion_matches = 0
        for triple_inst in combinations(triples, len(premise_var)):
            # 检查是否能变量化匹配
            entity_map = {}
            match = True
            for (ph, pr, pt), (th, tr, tt) in zip(premise_var, triple_inst):
                if pr != tr:
                    match = False
                    break
                # 变量绑定
                if ph not in entity_map:
                    entity_map[ph] = th
                elif entity_map[ph] != th:
                    match = False
                    break
                if pt not in entity_map:
                    entity_map[pt] = tt
                elif entity_map[pt] != tt:
                    match = False
                    break
            if not match:
                continue
            premise_matches += 1
            # 检查结论是否存在
            ch, cr, ct = conclusion_var
            ch_real = entity_map.get(ch, None)
            ct_real = entity_map.get(ct, None)
            if ch_real is None or ct_real is None:
                continue
            conclusion_inst = (ch_real, cr, ct_real)
            conclusion_in_kg = conclusion_inst in triples_set
            if conclusion_in_kg:
                conclusion_matches += 1
            # 记录实例
            rule_instances[(premise_var, conclusion_var)].append({
                'premise_instance': [tr for tr in triple_inst],
                'conclusion_instance': conclusion_inst,
                'conclusion_in_kg': conclusion_in_kg
            })
        confidence = conclusion_matches / premise_matches if premise_matches > 0 else 0
        if confidence >= RULE_MIN_CONFIDENCE:
            rule_counter[(premise_var, conclusion_var)] = (premise_matches, conclusion_matches, confidence)
    return rule_counter, rule_instances

# 4. 划分train/valid/test
def split_rules(rule_keys):
    random.shuffle(rule_keys)
    n = len(rule_keys)
    n_train = int(n * SPLIT_RATIO[0])
    n_valid = int(n * SPLIT_RATIO[1])
    train = rule_keys[:n_train]
    valid = rule_keys[n_train:n_train+n_valid]
    test = rule_keys[n_train+n_valid:]
    return train, valid, test

# 5. 保存规则和实例
def save_rules_and_instances(rule_keys, rule_counter, rule_instances, split_name):
    rules = []
    instances = []
    for idx, key in enumerate(rule_keys):
        premise_var, conclusion_var = key
        rule_id = f"rule_{split_name}_{idx+1:04d}"
        rules.append({
            'rule_id': rule_id,
            'premise': [ {'h': h, 'r': r, 't': t} for (h, r, t) in premise_var ],
            'conclusion': {'h': conclusion_var[0], 'r': conclusion_var[1], 't': conclusion_var[2]},
            'support': rule_counter[key][0],
            'conclusion_count': rule_counter[key][1],
            'confidence': rule_counter[key][2]
        })
        for inst in rule_instances[key]:
            instances.append({
                'rule_id': rule_id,
                'premise_instance': inst['premise_instance'],
                'conclusion_instance': inst['conclusion_instance'],
                'conclusion_in_kg': inst['conclusion_in_kg']
            })
    with open(os.path.join(OUTPUT_DIR, f'wn18rr_rules_{split_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(rules, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUTPUT_DIR, f'wn18rr_rule_instances_{split_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(instances, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print('加载三元组...')
    triples = load_triples()
    print(f'三元组总数: {len(triples)}')
    print('挖掘规则...')
    rule_counter, rule_instances = mine_rules(triples)
    print(f'高质量规则数: {len(rule_counter)}')
    rule_keys = list(rule_counter.keys())
    train, valid, test = split_rules(rule_keys)
    print(f'划分: train={len(train)}, valid={len(valid)}, test={len(test)}')
    print('保存train...')
    save_rules_and_instances(train, rule_counter, rule_instances, 'train')
    print('保存valid...')
    save_rules_and_instances(valid, rule_counter, rule_instances, 'valid')
    print('保存test...')
    save_rules_and_instances(test, rule_counter, rule_instances, 'test')
    print('全部完成！') 