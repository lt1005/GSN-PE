# main.py
import torch
from config import Config
from data_loader import DataLoader
from embedding import EmbeddingModel
from matcher import Matcher
from evaluation import evaluate

def main():
    config = Config()
    device = torch.device(config.device)

    # 数据加载
    data_loader = DataLoader(config)
    train_loader = data_loader.get_train_loader()
    eval_loader = data_loader.get_eval_loader()

    # 模型初始化
    embedding_model = EmbeddingModel(config).to(device)
    matcher = Matcher(config).to(device)

    # 优化器
    optimizer = torch.optim.Adam(list(embedding_model.parameters()) + list(matcher.parameters()), lr=config.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(config.epochs):
        embedding_model.train()
        matcher.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            subgraph_data = batch['subgraph_data'].to(device)
            rule_data = batch['rule_data'].to(device)
            labels = batch['label'].to(device)

            subgraph_emb = embedding_model(subgraph_data)
            rule_emb = embedding_model(rule_data)

            pred = matcher(subgraph_emb, rule_emb).squeeze()
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {total_loss/len(train_loader):.4f}")

        acc = evaluate(matcher, eval_loader, device)
        print(f"Validation Accuracy: {acc:.4f}")

if __name__ == '__main__':
    main()
