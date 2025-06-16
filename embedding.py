import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np

class PredicateEmbedding(nn.Module):
    """谓词嵌入层"""
    
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        
    def forward(self, predicate_indices):
        """
        输入: 谓词索引 (LongTensor)
        输出: 谓词嵌入向量
        """
        return self.embedding(predicate_indices)
    
    def aggregate_max(self, embeds):
        """
        对多个谓词嵌入进行max聚合
        """
        if len(embeds.shape) == 1:  # 单个谓词
            return embeds
        
        # 多个谓词，使用max聚合
        return torch.max(embeds, dim=0)[0]
    
    def aggregate_sum(self, embeds):
        """
        对多个谓词嵌入进行sum聚合
        """
        if len(embeds.shape) == 1:  # 单个谓词
            return embeds
        
        # 多个谓词，使用sum聚合
        return torch.sum(embeds, dim=0)

class RuleEncoder(nn.Module):
    """
    规则编码器：
    - 输入：规则谓词列表 (List[LongTensor]表示谓词索引)
    - motif_counts: 结构motif计数向量 (tensor)
    - 输出：规则嵌入向量，结构+语义融合
    """
    
    def __init__(self, predicate_vocab_size, predicate_embed_dim, motif_num, motif_embed_dim, output_dim):
        super().__init__()
        self.predicate_embedding = PredicateEmbedding(predicate_vocab_size, predicate_embed_dim)
        
        # motif结构编码 MLP
        self.motif_encoder = nn.Sequential(
            nn.Linear(motif_num, motif_embed_dim),
            nn.ReLU(),
            nn.Linear(motif_embed_dim, motif_embed_dim),
            nn.ReLU()
        )
        
        # 结构+语义融合层
        self.fusion = nn.Sequential(
            nn.Linear(predicate_embed_dim + motif_embed_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, predicate_indices_list, motif_counts_batch):
        """
        predicate_indices_list: List[LongTensor] 每个元素是当前规则的谓词索引 tensor([idx1, idx2, ...])
        motif_counts_batch: Tensor shape (batch_size, motif_num) motif计数
        
        返回：batch_size x output_dim 规则向量
        """
        
        batch_pred_embeds = []
        for predicate_indices in predicate_indices_list:
            embeds = self.predicate_embedding(predicate_indices)  # (num_predicates, embed_dim)
            # 这里用max聚合实现包含性，改成sum也行，但max更直观满足子集关系
            agg_embed = self.predicate_embedding.aggregate_max(embeds)  # (embed_dim,)
            batch_pred_embeds.append(agg_embed)
        
        predicate_batch = torch.stack(batch_pred_embeds, dim=0)  # (batch_size, predicate_embed_dim)
        
        motif_embeds = self.motif_encoder(motif_counts_batch)  # (batch_size, motif_embed_dim)
        
        fused = torch.cat([predicate_batch, motif_embeds], dim=-1)
        out = self.fusion(fused)  # (batch_size, output_dim)
        
        return out

class SubgraphEncoder(nn.Module):
    """
    子图编码器：
    - 输入：子图对象 (List[networkx.MultiDiGraph])
    - motif_counts: 结构motif计数向量 (tensor)
    - 输出：子图嵌入向量，结构+语义融合
    """
    
    def __init__(self, entity_vocab_size, predicate_vocab_size, embed_dim, motif_num, motif_embed_dim, output_dim):
        super().__init__()
        self.entity_embedding = nn.Embedding(entity_vocab_size, embed_dim)
        self.predicate_embedding = PredicateEmbedding(predicate_vocab_size, embed_dim)
        
        # motif结构编码 MLP
        self.motif_encoder = nn.Sequential(
            nn.Linear(motif_num, motif_embed_dim),
            nn.ReLU(),
            nn.Linear(motif_embed_dim, motif_embed_dim),
            nn.ReLU()
        )
        
        # 结构+语义融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + motif_embed_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, subgraphs, motif_counts_batch):
        """
        subgraphs: List[networkx.MultiDiGraph] 子图对象列表
        motif_counts_batch: Tensor shape (batch_size, motif_num) motif计数
        
        返回：batch_size x output_dim 子图向量
        """
        batch_embeds = []
        
        for i, subgraph in enumerate(subgraphs):
            # 检查子图是否为None
            if subgraph is None or subgraph.number_of_edges() == 0:
                # 创建零向量表示空图
                semantic_embed = torch.zeros(self.entity_embedding.embedding_dim).to(self.entity_embedding.weight.device)
                batch_embeds.append(semantic_embed)
                continue
            
            # 收集所有实体和谓词
            entity_indices = []
            predicate_indices = []
            
            # 遍历子图中的所有边
            for u, v, key in subgraph.edges(keys=True):
                # 假设u和v是实体索引，key是谓词索引
                entity_indices.append(u)
                entity_indices.append(v)
                predicate_indices.append(key)
            
            # 去重
            entity_indices = list(set(entity_indices))
            predicate_indices = list(set(predicate_indices))
            
            # 转换为tensor
            entity_indices = torch.tensor(entity_indices, dtype=torch.long).to(self.entity_embedding.weight.device)
            predicate_indices = torch.tensor(predicate_indices, dtype=torch.long).to(self.predicate_embedding.embedding.weight.device)
            
            # 获取嵌入
            entity_embeds = self.entity_embedding(entity_indices)  # (num_entities, embed_dim)
            predicate_embeds = self.predicate_embedding(predicate_indices)  # (num_predicates, embed_dim)
            
            # 聚合实体和谓词嵌入
            if len(entity_embeds) > 0:
                entity_embed = torch.mean(entity_embeds, dim=0)  # (embed_dim,)
            else:
                entity_embed = torch.zeros(self.entity_embedding.embedding_dim).to(self.entity_embedding.weight.device)
            
            if len(predicate_embeds) > 0:
                predicate_embed = torch.mean(predicate_embeds, dim=0)  # (embed_dim,)
            else:
                predicate_embed = torch.zeros(self.predicate_embedding.embed_dim).to(self.predicate_embedding.embedding.weight.device)
            
            # 合并实体和谓词嵌入
            semantic_embed = (entity_embed + predicate_embed) / 2.0  # (embed_dim,)
            batch_embeds.append(semantic_embed)
        
        semantic_batch = torch.stack(batch_embeds, dim=0)  # (batch_size, embed_dim)
        
        motif_embeds = self.motif_encoder(motif_counts_batch)  # (batch_size, motif_embed_dim)
        
        fused = torch.cat([semantic_batch, motif_embeds], dim=-1)
        out = self.fusion(fused)  # (batch_size, output_dim)
        
        return out
