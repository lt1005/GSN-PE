# config.py
class Config:
    def __init__(self):
        # 模型超参数
        self.batch_size = 32
        self.lr = 1e-3
        self.epochs = 20
        self.embedding_dim = 128
        self.max_hop = 2

        # 设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 数据文件路径
        self.rules_path = './rules/sample_rules.json'
        self.kg_path = './rules/sample_kg.edgelist'

        # 数据集划分比例
        self.train_ratio = 0.8
