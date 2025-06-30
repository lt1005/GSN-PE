"""
# StructMatch: 基于规则结构驱动的子图扩展与匹配系统

## 项目简介

StructMatch是一个基于结构规则驱动的知识图谱子图匹配与推理系统。系统利用结构嵌入技术，能够从给定的anchor三元组开始，智能扩展子图并判断是否符合预定义的结构规则（如T型、Y型、Fork等）。

## 核心特性

- **双模结构嵌入**: 结合motif-count向量和GNN嵌入，提供丰富的结构表示
- **智能子图扩展**: 基于结构相似度的beam search扩展策略
- **灵活规则匹配**: 支持多种结构模式（三角形、星形、路径等）
- **可解释性**: 提供详细的匹配得分和结构分析

## 安装依赖

```bash
pip install torch>=2.0.0
pip install torch_geometric>=2.3.0
pip install networkx>=3.0
pip install scikit-learn>=1.3.0
pip install numpy>=1.24.0
pip install tqdm>=4.65.0
```

## 数据格式说明

### 知识图谱格式 (example_kg.txt)
```
实体1	关系	实体2
A	friendOf	B
B	worksWith	C
```

### 规则格式 (rules.json)
```json
[
  {
    "id": 1,
    "name": "Triangle Pattern",
    "premise": {
      "structure_type": "Triangle",
      "description": "三角形结构"
    },
    "conclusion": "Strong connection"
  }
]
```

## 快速开始

```python
from main import StructMatchSystem
from utils.config import Config

# 初始化系统
config = Config()
system = StructMatchSystem(config)

# 加载数据
system.load_data("data/example_kg.txt", "data/rules.json")

# 定义测试anchor
anchors = [("A", "friendOf", "B")]

# 运行匹配
results = system.run_batch_matching(anchors, rule_idx=0)

# 查看结果
for result in results:
    print(f"匹配成功: {result['is_match']}")
    print(f"得分: {result['match_score']:.4f}")
```

## 模块说明

### 数据加载模块 (data/)
- `kg_loader.py`: 加载知识图谱三元组
- `rule_loader.py`: 加载结构规则定义

### 结构嵌入模块 (struct_embed/)
- `motif_extractor.py`: 提取图motif特征向量
- `gnn_encoder.py`: GNN结构嵌入编码器
- `embed_fusion.py`: 多模态嵌入融合

### 匹配模块 (matcher/)
- `graph_expander.py`: 智能子图扩展器
- `struct_matcher.py`: 结构模式匹配器
- `scoring.py`: 相似度计算函数

### 训练模块 (trainer/)
- `train_gnn.py`: GNN模型训练器
- `contrastive_loss.py`: 对比学习损失函数

## 运行示例

```bash
python main.py
```

这将：
1. 创建示例数据文件
2. 加载知识图谱和规则
3. 运行结构匹配
4. 输出匹配结果并保存到results.json

## 配置参数

可以通过修改`Config`类来调整系统参数：

```python
@dataclass
class Config:
    motif_dim: int = 8              # motif特征维度
    gnn_hidden_dim: int = 64        # GNN隐藏层维度
    fusion_dim: int = 40            # 融合嵌入维度
    max_expand_steps: int = 5       # 最大扩展步数
    similarity_threshold: float = 0.8 # 匹配阈值
```

## 扩展功能

系统支持以下扩展：
- 自定义结构规则定义
- 新的motif类型添加  
- 不同的GNN架构
- 可视化工具集成

## 注意事项

1. 确保知识图谱文件格式正确（tab分隔的三元组）
2. 规则文件必须是有效的JSON格式
3. 大型图数据建议调整beam_size和max_expand_steps参数
4. GPU可用时会自动使用CUDA加速

## 许可证

MIT License
"""