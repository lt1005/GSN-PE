import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

class GNNTrainer:
    """GNN结构嵌入训练器"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate
        )
        self.criterion = ContrastiveLoss(temperature=0.1)
        
        # 移动模型到设备
        self.model.to(self.device)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)
            
            # 前向传播
            embeddings = self.model(batch)
            labels = batch.y if hasattr(batch, 'y') else torch.zeros(batch.batch.max() + 1)
            labels = labels.to(self.device)
            
            # 计算损失
            loss = self.criterion(embeddings, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, 
              val_loader: DataLoader = None) -> Dict[str, List[float]]:
        """完整训练过程"""
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config.num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                logging.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                logging.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
        
        return history
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                embeddings = self.model(batch)
                labels = batch.y if hasattr(batch, 'y') else torch.zeros(batch.batch.max() + 1)
                labels = labels.to(self.device)
                
                loss = self.criterion(embeddings, labels)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
