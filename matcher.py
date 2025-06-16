import torch
import torch.nn as nn
import torch.nn.functional as F

class AnchorRuleMatcher(nn.Module):
    """规则与子图匹配器"""
    
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        # 可加一些额外层进行嵌入对齐或融合（如果需要）
        self.classifier = nn.Linear(embedding_dim * 2, 1)  # 输出相似度/匹配得分
    
    def forward(self, subgraph_emb, rule_emb):
        """
        输入:
            subgraph_emb: [batch_size, embedding_dim] 知识图谱实例子图嵌入
            rule_emb: [batch_size, embedding_dim] 规则图嵌入
        输出:
            match_score: [batch_size, 1] 匹配得分，越大表示越匹配
        """
        combined = torch.cat([subgraph_emb, rule_emb], dim=-1)
        score = torch.sigmoid(self.classifier(combined))
        return score
    
    def match(self, subgraph_emb, rule_emb, threshold=0.5):
        """
        判断是否匹配
        """
        score = self.forward(subgraph_emb, rule_emb)
        return (score >= threshold).float(), score

def match_inference(matcher, subgraph_emb, rule_emb, threshold=0.5):
    """
    外部调用接口，方便推理时用
    """
    matcher.eval()
    with torch.no_grad():
        matched, score = matcher.match(subgraph_emb, rule_emb, threshold)
    return matched, score
