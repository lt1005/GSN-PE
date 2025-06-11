# evaluation.py
import torch
import torch.nn.functional as F

def compute_accuracy(pred_scores, labels, threshold=0.5):
    preds = (pred_scores >= threshold).float()
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0

def evaluate(matcher, dataloader, device):
    matcher.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            subgraph_emb = batch['subgraph_emb'].to(device)
            rule_emb = batch['rule_emb'].to(device)
            labels = batch['label'].to(device)
            scores = matcher(subgraph_emb, rule_emb).squeeze()
            all_scores.append(scores)
            all_labels.append(labels)
    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)
    acc = compute_accuracy(all_scores, all_labels)
    return acc
