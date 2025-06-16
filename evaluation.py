import torch
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score 

def evaluate(trainer, data_loader, device):
    """评估模型性能"""
    # 将模型设置为评估模式
    trainer.rule_encoder.eval()
    trainer.subgraph_encoder.eval()
    trainer.matcher.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            premises_list = [p.to(device) for p in batch['premises_list']]
            premise_motif_counts = batch['premise_motif_counts'].to(device)
            positive_subgraphs = batch['positive_subgraphs']
            positive_motif_counts = batch['positive_motif_counts'].to(device)
            negative_subgraphs = batch['negative_subgraphs']
            negative_motif_counts = batch['negative_motif_counts'].to(device)
            
            # 计算正样本得分
            rule_emb = trainer.rule_encoder(premises_list, premise_motif_counts)
            positive_emb = trainer.subgraph_encoder(positive_subgraphs, positive_motif_counts)
            positive_scores = trainer.matcher(rule_emb, positive_emb)
            
            # 计算负样本得分
            negative_emb = trainer.subgraph_encoder(negative_subgraphs, negative_motif_counts)
            negative_scores = trainer.matcher(rule_emb, negative_emb)
            
            # 收集得分和标签
            batch_scores = torch.cat([positive_scores, negative_scores], dim=0).squeeze().cpu().numpy()
            batch_labels = torch.cat([
                torch.ones_like(positive_scores),
                torch.zeros_like(negative_scores)
            ], dim=0).squeeze().cpu().numpy()
            
            all_scores.extend(batch_scores)
            all_labels.extend(batch_labels)
    
    # 计算评估指标
    if len(set(all_labels)) < 2:
        # 处理标签全为0或全为1的情况
        auc = 0.5  # 随机猜测的AUC值
    else:
        try:
            auc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            auc = 0.5  # 处理无法计算AUC的情况
    
    accuracy = np.mean((np.array(all_scores) > 0.5) == all_labels)
    precision = precision_score(all_labels, np.array(all_scores) > 0.5) if len(set(all_labels)) > 1 else 0.0
    recall = recall_score(all_labels, np.array(all_scores) > 0.5) if len(set(all_labels)) > 1 else 0.0
    f1 = f1_score(all_labels, np.array(all_scores) > 0.5) if len(set(all_labels)) > 1 else 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc  # 添加AUC到指标字典
    }

def evaluate_subgraph_isomorphism(model, data_loader, device):
    """
    评估子图同构检测性能
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
    
    返回:
        评估指标字典
    """
    model.eval()
    
    all_correct = []
    all_total = []
    
    with torch.no_grad():
        for batch in data_loader:
            premises_list = [p.to(device) for p in batch['premises_list']]
            premise_motif_counts = batch['premise_motif_counts'].to(device)
            positive_subgraphs = batch['positive_subgraphs']
            positive_motif_counts = batch['positive_motif_counts'].to(device)
            
            # 编码规则
            rule_embeddings = model.rule_encoder(premises_list, premise_motif_counts)
            
            # 编码正样本子图
            positive_subgraph_embeddings = model.subgraph_encoder(
                positive_subgraphs, 
                positive_motif_counts
            )
            
            # 计算匹配得分
            positive_scores = model.matcher(positive_subgraph_embeddings, rule_embeddings)
            
            # 计算每个规则的子图同构检测准确率
            for i, score in enumerate(positive_scores):
                # 这里简化处理，假设得分越高表示越匹配
                is_matched = (score >= 0.5).item()
                
                # 真实标签（假设正样本都是匹配的）
                true_label = True
                
                all_correct.append(is_matched == true_label)
                all_total.append(1)
    
    # 计算准确率
    isomorphism_accuracy = sum(all_correct) / sum(all_total)
    
    metrics = {
        'isomorphism_accuracy': isomorphism_accuracy
    }
    
    return metrics
