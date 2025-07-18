import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Tuple, Optional
import logging

class SemanticFeatureEnhancer:
    """语义特征增强模块 - 用KG embedding作为特征，增强结构匹配的语义区分能力"""
    
    def __init__(self, embedding_dim: int = 100, feature_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.logger = logging.getLogger(__name__)
        
        # 实体和关系的embedding字典（从TransE等预训练模型加载）
        self.entity_embeddings = {}
        self.relation_embeddings = {}
        
        # 特征转换网络 - 将embedding转换为适合融合的特征
        self.entity_feature_net = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.relation_feature_net = nn.Sequential(
            nn.Linear(embedding_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 初始化embedding（如果没有预训练文件，用随机初始化）
        self._init_embeddings()
    
    def _init_embeddings(self):
        """初始化embedding，优先加载预训练文件，否则随机初始化"""
        # 尝试加载预训练的embedding文件
        if self._load_pretrained_embeddings():
            self.logger.info("成功加载预训练embedding")
        else:
            self.logger.info("使用随机初始化的embedding")
            self._init_random_embeddings()
    
    def _load_pretrained_embeddings(self) -> bool:
        """尝试加载预训练的embedding文件"""
        try:
            # 检查是否存在embedding文件
            entity_file = "embeddings/entity_embeddings.pkl"
            relation_file = "embeddings/relation_embeddings.pkl"
            
            if os.path.exists(entity_file) and os.path.exists(relation_file):
                # 加载pickle格式的embedding
                import pickle
                with open(entity_file, 'rb') as f:
                    entity_embeds = pickle.load(f)
                with open(relation_file, 'rb') as f:
                    relation_embeds = pickle.load(f)
                
                self.entity_embeddings = entity_embeds
                self.relation_embeddings = relation_embeds
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.warning(f"加载预训练embedding失败: {e}")
            return False
    
    def _init_random_embeddings(self):
        """随机初始化embedding"""
        # 从训练集中读取实体和关系
        entities = set()
        relations = set()
        
        try:
            with open("dataseet/wn18rr/train.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    h, r, t = line.strip().split('\t')
                    entities.add(h)
                    entities.add(t)
                    relations.add(r)
            
            # 随机初始化embedding
            for entity in entities:
                self.entity_embeddings[entity] = np.random.normal(
                    0, 0.1, self.embedding_dim
                ).astype(np.float32)
            
            for relation in relations:
                self.relation_embeddings[relation] = np.random.normal(
                    0, 0.1, self.embedding_dim
                ).astype(np.float32)
                
        except Exception as e:
            self.logger.error(f"初始化embedding失败: {e}")
    
    def get_entity_feature(self, entity: str) -> torch.Tensor:
        """获取实体的语义特征"""
        try:
            # 获取embedding
            if entity in self.entity_embeddings:
                embed = torch.tensor(self.entity_embeddings[entity], dtype=torch.float32)
            else:
                # 如果实体不存在，使用零向量
                embed = torch.zeros(self.embedding_dim, dtype=torch.float32)
            
            # 转换为特征向量
            feature = self.entity_feature_net(embed)
            return feature
            
        except Exception as e:
            self.logger.error(f"获取实体特征失败: {e}")
            return torch.zeros(self.feature_dim, dtype=torch.float32)
    
    def get_relation_feature(self, relation: str) -> torch.Tensor:
        """获取关系的语义特征"""
        try:
            # 获取embedding
            if relation in self.relation_embeddings:
                embed = torch.tensor(self.relation_embeddings[relation], dtype=torch.float32)
            else:
                # 如果关系不存在，使用零向量
                embed = torch.zeros(self.embedding_dim, dtype=torch.float32)
            
            # 转换为特征向量
            feature = self.relation_feature_net(embed)
            return feature
            
        except Exception as e:
            self.logger.error(f"获取关系特征失败: {e}")
            return torch.zeros(self.feature_dim, dtype=torch.float32)
    
    def get_subgraph_semantic_features(self, subgraph) -> torch.Tensor:
        """获取子图的语义特征（聚合所有实体和关系的特征）"""
        try:
            entity_features = []
            relation_features = []
            
            # 提取实体特征
            for node in subgraph.nodes():
                feature = self.get_entity_feature(node)
                entity_features.append(feature)
            
            # 提取关系特征
            for _, _, edge_data in subgraph.edges(data=True):
                relation = edge_data.get('relation', 'unknown')
                feature = self.get_relation_feature(relation)
                relation_features.append(feature)
            
            # 聚合特征
            all_features = entity_features + relation_features
            if all_features:
                # 使用平均池化
                semantic_feature = torch.stack(all_features).mean(dim=0)
            else:
                semantic_feature = torch.zeros(self.feature_dim, dtype=torch.float32)
            
            return semantic_feature
            
        except Exception as e:
            self.logger.error(f"获取子图语义特征失败: {e}")
            return torch.zeros(self.feature_dim, dtype=torch.float32)
    
    def semantic_similarity(self, feature1: torch.Tensor, feature2: torch.Tensor) -> float:
        """计算两个语义特征的相似度"""
        try:
            # 使用余弦相似度
            similarity = F.cosine_similarity(
                feature1.unsqueeze(0), feature2.unsqueeze(0), dim=1
            )
            return similarity.item()
        except Exception as e:
            self.logger.error(f"计算语义相似度失败: {e}")
            return 0.0

class EnhancedSemanticVerifier:
    """增强的语义验证器 - 结合结构特征和语义特征"""
    
    def __init__(self, semantic_enhancer: SemanticFeatureEnhancer, 
                 threshold: float = 0.7, feature_dim: int = 64):
        self.semantic_enhancer = semantic_enhancer
        self.threshold = threshold
        self.feature_dim = feature_dim
        self.logger = logging.getLogger(__name__)
        
        # 特征融合网络 - 融合结构特征和语义特征
        # 注意：结构特征维度是40，语义特征维度是64
        self.fusion_net = nn.Sequential(
            nn.Linear(40 + feature_dim, 128),  # 结构(40) + 语义(64) = 104 -> 128
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def semantic_verify(self, subgraph, rule_semantic_vec: torch.Tensor) -> Tuple[bool, float]:
        """语义验证 - 兼容接口方法"""
        try:
            # 获取子图的语义特征
            subgraph_semantic_feature = self.semantic_enhancer.get_subgraph_semantic_features(subgraph)
            
            # 计算语义相似度
            semantic_score = self.semantic_enhancer.semantic_similarity(
                subgraph_semantic_feature, rule_semantic_vec
            )
            
            # 判断是否通过验证
            is_passed = semantic_score >= self.threshold
            
            return is_passed, semantic_score
            
        except Exception as e:
            self.logger.error(f"语义验证失败: {e}")
            return False, 0.0
    
    def verify_with_enhanced_semantics(self, subgraph, rule_semantic_vec: torch.Tensor,
                                     struct_feature: torch.Tensor) -> Tuple[bool, float]:
        """增强语义验证 - 结合结构特征和语义特征"""
        try:
            # 获取子图的语义特征
            subgraph_semantic_feature = self.semantic_enhancer.get_subgraph_semantic_features(subgraph)
            
            # 确保结构特征是tensor格式
            if not isinstance(struct_feature, torch.Tensor):
                if isinstance(struct_feature, (list, tuple)):
                    struct_feature = torch.tensor(struct_feature, dtype=torch.float32)
                else:
                    struct_feature = torch.tensor([struct_feature], dtype=torch.float32)
            
            # 确保语义特征是tensor格式
            if not isinstance(subgraph_semantic_feature, torch.Tensor):
                if isinstance(subgraph_semantic_feature, (list, tuple)):
                    subgraph_semantic_feature = torch.tensor(subgraph_semantic_feature, dtype=torch.float32)
                else:
                    subgraph_semantic_feature = torch.tensor([subgraph_semantic_feature], dtype=torch.float32)
            
            # 确保维度匹配
            if struct_feature.dim() == 0:
                struct_feature = struct_feature.unsqueeze(0)
            if subgraph_semantic_feature.dim() == 0:
                subgraph_semantic_feature = subgraph_semantic_feature.unsqueeze(0)
            
            # 检查特征维度
            if struct_feature.size(0) != 40:
                # 如果结构特征不是40维，使用零填充或截断
                if struct_feature.size(0) < 40:
                    padding = torch.zeros(40 - struct_feature.size(0), dtype=torch.float32)
                    struct_feature = torch.cat([struct_feature, padding])
                else:
                    struct_feature = struct_feature[:40]
            
            if subgraph_semantic_feature.size(0) != self.feature_dim:
                # 如果语义特征维度不匹配，调整
                if subgraph_semantic_feature.size(0) < self.feature_dim:
                    padding = torch.zeros(self.feature_dim - subgraph_semantic_feature.size(0), dtype=torch.float32)
                    subgraph_semantic_feature = torch.cat([subgraph_semantic_feature, padding])
                else:
                    subgraph_semantic_feature = subgraph_semantic_feature[:self.feature_dim]
            
            # 融合结构特征和语义特征
            combined_feature = torch.cat([struct_feature, subgraph_semantic_feature], dim=0)
            
            # 通过融合网络
            with torch.no_grad():
                fusion_score = self.fusion_net(combined_feature.unsqueeze(0)).item()
            
            # 判断是否通过验证
            is_passed = fusion_score >= self.threshold
            
            return is_passed, fusion_score
            
        except Exception as e:
            self.logger.error(f"增强语义验证失败: {e}")
            return False, 0.0 