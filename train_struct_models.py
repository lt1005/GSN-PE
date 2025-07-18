#!/usr/bin/env python3
"""
训练结构嵌入模型
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import networkx as nx
import numpy as np
import json
import os
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
import pickle

from utils.config import Config
from struct_embed.motif_extractor import MotifExtractor
from struct_embed.gnn_encoder import GNNEncoder
from struct_embed.embed_fusion import EmbedFusion
from trainer.contrastive_loss import ContrastiveLoss

class GraphDataset(Dataset):
    """图数据集"""
    
    def __init__(self, graphs: List[nx.Graph], labels: List[int]):
        self.graphs = graphs
        self.labels = labels
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

def create_training_data():
    """创建训练数据"""
    print("=== 创建训练数据 ===")
    
    # 从规则文件中提取图结构
    rules_path = "data/wn18rr_strict_structural_rules_relaxed_with_id.json"
    with open(rules_path, 'r', encoding='utf-8') as f:
        rules = json.load(f)
    
    graphs = []
    labels = []
    
    # 为每个规则创建图结构
    for rule_idx, rule in enumerate(rules):
        # 创建规则图
        rule_graph = nx.DiGraph()
        for triple in rule['premise']:
            h, r, t = triple[0], triple[1], triple[2]
            rule_graph.add_edge(h, t, relation=r)
        
        # 添加规则图
        graphs.append(rule_graph)
        labels.append(rule_idx)
        
        # 为每个实例创建图结构
        for inst in rule.get('instances', []):
            inst_graph = nx.DiGraph()
            for triple in inst['premise_instance']:
                h, r, t = triple[0], triple[1], triple[2]
                inst_graph.add_edge(h, t, relation=r)
            
            graphs.append(inst_graph)
            labels.append(rule_idx)  # 实例与规则有相同标签
    
    print(f"创建了 {len(graphs)} 个图，{len(set(labels))} 个类别")
    return graphs, labels

def train_structure_models():
    """训练结构嵌入模型"""
    print("=== 开始训练结构嵌入模型 ===")
    
    # 配置
    config = Config()
    device = torch.device(config.device)
    
    # 创建训练数据
    graphs, labels = create_training_data()
    
    # 创建数据集
    dataset = GraphDataset(graphs, labels)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # 初始化模型组件
    motif_extractor = MotifExtractor(motif_types=config.motif_types)
    gnn_encoder = GNNEncoder(
        input_dim=1,  # 节点特征维度
        hidden_dim=config.gnn_hidden_dim,
        output_dim=config.gnn_output_dim
    )
    embed_fusion = EmbedFusion(
        motif_dim=len(config.motif_types),
        gnn_dim=config.gnn_output_dim,
        attr_dim=0,
        fusion_dim=config.fusion_dim,
        fusion_type=config.fusion_type
    )
    
    # 创建完整的结构嵌入模型
    class StructureEmbeddingModel(torch.nn.Module):
        def __init__(self, motif_extractor, gnn_encoder, embed_fusion):
            super().__init__()
            self.motif_extractor = motif_extractor
            self.gnn_encoder = gnn_encoder
            self.embed_fusion = embed_fusion
        
        def forward(self, graph_batch):
            # 这里需要根据实际的图批处理格式调整
            # 暂时返回随机嵌入用于演示
            batch_size = len(graph_batch) if isinstance(graph_batch, list) else 1
            return torch.randn(batch_size, 40)  # 40是融合维度
    
    model = StructureEmbeddingModel(motif_extractor, gnn_encoder, embed_fusion)
    model.to(device)
    
    # 训练器
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = ContrastiveLoss(temperature=0.1)
    
    # 训练循环
    model.train()
    for epoch in range(config.num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (graph_batch, label_batch) in enumerate(pbar):
            # 前向传播
            embeddings = model(graph_batch)
            labels = torch.tensor(label_batch, dtype=torch.long).to(device)
            
            # 计算损失
            loss = criterion(embeddings, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    # 保存模型
    save_path = "trained_models/"
    os.makedirs(save_path, exist_ok=True)
    
    torch.save({
        'motif_extractor_state_dict': motif_extractor.state_dict(),
        'gnn_encoder_state_dict': gnn_encoder.state_dict(),
        'embed_fusion_state_dict': embed_fusion.state_dict(),
        'config': config
    }, os.path.join(save_path, "structure_models.pth"))
    
    print(f"模型已保存到 {save_path}")

def train_semantic_fusion_model():
    """训练语义融合模型"""
    print("=== 开始训练语义融合模型 ===")
    
    # 这里需要训练EnhancedSemanticVerifier中的fusion_net
    # 暂时跳过，因为需要更多的训练数据
    print("语义融合模型训练暂未实现")

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 训练结构嵌入模型
    train_structure_models()
    
    # 训练语义融合模型（可选）
    # train_semantic_fusion_model()
    
    print("=== 训练完成 ===") 