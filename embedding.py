import torch
import torch.nn as nn
import torch.nn.functional as F

class PredicateEmbedding(nn.Module):
    """
    谓词基向量，强制非负，支持集合最大/加和，保证无序性和包含性。
    """
    def __init__(self, predicate_vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(predicate_vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.embed.weight)
        # 用ReLU保证非负
        self.non_negative = nn.ReLU()

    def forward(self, predicate_indices):
        # predicate_indices: LongTensor shape (num_predicates,)
        raw_embeds = self.embed(predicate_indices)  # (num_predicates, embed_dim)
        embeds = self.non_negative(raw_embeds)
        return embeds

    def aggregate_max(self, embeds):
        # 根据max操作实现子集包含性质 (无序性保证)
        # embeds: (num_predicates, embed_dim)
        return torch.max(embeds, dim=0)[0]

    def aggregate_sum(self, embeds):
        # 加法聚合，实现可加性（也可尝试max + sum结合）
        return torch.sum(embeds, dim=0)


class RuleEncoder(nn.Module):
    """
    规则编码器：
    - 输入：规则谓词列表 (List[int]表示谓词索引)
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
            nn.ReLU(),
        )

        # 结构+语义融合层
        self.fusion = nn.Sequential(
            nn.Linear(predicate_embed_dim + motif_embed_dim, output_dim),
            nn.ReLU(),
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
    子图编码器（对实例子图编码）
    这里假设你有一个预训练/自定义的GNN模块
    """

    def __init__(self, gnn_model, motif_num, motif_embed_dim, output_dim):
        """
        gnn_model: 传入的图神经网络实例，输出固定维度 gnn_out_dim
        """
        super().__init__()
        self.gnn_model = gnn_model  # 例如GNNEncoder实例

        self.motif_encoder = nn.Sequential(
            nn.Linear(motif_num, motif_embed_dim),
            nn.ReLU(),
            nn.Linear(motif_embed_dim, motif_embed_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(gnn_model.output_dim + motif_embed_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, node_feats, edge_index, motif_counts, batch=None):
        """
        node_feats: [N, node_feat_dim]
        edge_index: [2, E]
        motif_counts: [batch_size, motif_num]
        batch: [N] 指定节点所属图index
        """

        semantic_emb = self.gnn_model(node_feats, edge_index, batch)  # (batch_size, gnn_out_dim)
        motif_emb = self.motif_encoder(motif_counts)  # (batch_size, motif_embed_dim)
        combined = torch.cat([semantic_emb, motif_emb], dim=-1)
        out = self.fusion(combined)  # (batch_size, output_dim)
        return out

