import argparse
import torch
import os
import json
from data_loader import RuleDataset, load_relations, load_entities, load_mined_rules, collate_fn
from trainer import RuleAlignmentTrainer
from evaluation import evaluate, evaluate_subgraph_isomorphism
from torch.utils.data import DataLoader
import numpy as np

def set_seed(seed):
    """设置随机种子，保证结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='GSN-PE 规则推理训练')
    parser.add_argument('--rules_file', type=str, required=True, help='挖掘规则文件路径 (mined_rules.txt)')
    parser.add_argument('--entities_dict', type=str, required=True, help='实体字典路径 (entities.dict)')
    parser.add_argument('--relations_dict', type=str, required=True, help='关系字典路径 (relations.txt)')
    parser.add_argument('--train_file', type=str, required=True, help='训练集三元组路径 (train.txt)')
    parser.add_argument('--valid_file', type=str, required=True, help='验证集三元组路径 (valid.txt)')
    parser.add_argument('--test_file', type=str, required=True, help='测试集三元组路径 (test.txt)')
    parser.add_argument('--output_dir', type=str, default='./output', help='模型输出目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--embed_dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--max_subgraph_size', type=int, default=8, help='最大子图大小')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备 (cpu/cuda)')
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # 加载字典文件
    entity_vocab = load_entities(args.entities_dict)
    predicate_vocab = load_relations(args.relations_dict)
    print(f"加载实体字典: {len(entity_vocab)} 个实体")
    print(f"加载关系字典: {len(predicate_vocab)} 种关系")

    # 定义Motif模板（示例模板，需根据任务调整）
    motif_templates = [
        ['p1', 'p2'],          # 二元关系链
        ['p1', 'p3', 'p2'],    # 三元关系链
        ['p1', 'p2', 'p1']     # 对称关系
    ]

    # 加载数据集
    train_dataset = RuleDataset(
        rules_file=args.rules_file,
        kg_files=[args.train_file],
        motif_templates=motif_templates,
        predicate_vocab=predicate_vocab,
        entity_vocab=entity_vocab,
        max_subgraph_size=args.max_subgraph_size
    )
    valid_dataset = RuleDataset(
        rules_file=args.rules_file,
        kg_files=[args.valid_file],
        motif_templates=motif_templates,
        predicate_vocab=predicate_vocab,
        entity_vocab=entity_vocab,
        max_subgraph_size=args.max_subgraph_size
    )
    test_dataset = RuleDataset(
        rules_file=args.rules_file,
        kg_files=[args.test_file],
        motif_templates=motif_templates,
        predicate_vocab=predicate_vocab,
        entity_vocab=entity_vocab,
        max_subgraph_size=args.max_subgraph_size
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    config = {
        'entity_num': len(entity_vocab),
        'relation_num': len(predicate_vocab),
        'predicate_vocab_size': len(predicate_vocab),
        'entity_vocab_size': len(entity_vocab),
        'embed_dim': args.embed_dim,
        'motif_num': len(motif_templates),
        'motif_embed_dim': 64,
        'output_dim': 128, 
        'batch_size': args.batch_size,
        'learning_rate': args.lr, 
        'weight_decay': 0.0001, 
        'max_subgraph_size': args.max_subgraph_size,
        'device': args.device
    }
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 初始化训练器
    trainer = RuleAlignmentTrainer(config)
    trainer.load_data(train_dataset, valid_dataset)

    # 训练模型
    best_valid_f1 = 0.0
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch()
        valid_metrics = evaluate(trainer, valid_loader, args.device)
        
        # 打印验证集结果
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f}")
        print(f"Valid F1: {valid_metrics['f1']:.4f} | Precision: {valid_metrics['precision']:.4f} | Recall: {valid_metrics['recall']:.4f}")
        
        # 保存最佳模型
        if valid_metrics['f1'] > best_valid_f1:
            best_valid_f1 = valid_metrics['f1']
            torch.save({
                'rule_encoder_state': trainer.rule_encoder.state_dict(),
                'subgraph_encoder_state': trainer.subgraph_encoder.state_dict(),
                'matcher_state': trainer.matcher.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'metrics': valid_metrics
            }, best_model_path)
            print(f"★ 保存最佳模型，F1: {best_valid_f1:.4f}")

    # 加载最佳模型并评估测试集
    checkpoint = torch.load(best_model_path)
    trainer.rule_encoder.load_state_dict(checkpoint['rule_encoder_state'])
    trainer.subgraph_encoder.load_state_dict(checkpoint['subgraph_encoder_state'])
    trainer.matcher.load_state_dict(checkpoint['matcher_state'])
    test_metrics = evaluate(trainer, test_loader, args.device)
    iso_metrics = evaluate_subgraph_isomorphism(trainer, test_loader, args.device)

    # 输出最终结果
    print("\n=== 测试集评估结果 ===")
    print(f"F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['auc']:.4f}")
    print(f"子图同构准确率: {iso_metrics['isomorphism_acc']:.4f}")
    
    # 保存结果
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'classification': test_metrics,
            'isomorphism': iso_metrics
        }, f, indent=2)
    print(f"结果已保存至 {args.output_dir}")

if __name__ == "__main__":
    main()
