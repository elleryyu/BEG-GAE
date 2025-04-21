import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

# 导入您的 GCN 模型和工具函数
from model import GCN  # 假设 GCN 模型定义在 model.py 中
from utils import set_random_seed

def train_epoch(model, optimizer, device, graph):
    model.train()
    graph = graph.to(device)
    optimizer.zero_grad()

    features = graph.ndata['feature']
    train_mask = graph.ndata['train_mask']

    # 前向传播
    reconstructed, embeddings = model(graph, features)

    # 只计算训练集上的损失
    loss = F.mse_loss(reconstructed[train_mask], features[train_mask])
    loss.backward()
    optimizer.step()

    epoch_loss = loss.detach().item()
    return epoch_loss

def evaluate(model, graph, device):
    model.eval()
    graph = graph.to(device)
    with torch.no_grad():
        features = graph.ndata['feature']
        val_mask = graph.ndata['val_mask']

        reconstructed, _ = model(graph, features)

        # 只计算验证集上的损失
        loss = F.mse_loss(reconstructed[val_mask], features[val_mask])
    return loss.item()

def save_model(model, optimizer, epoch, best_loss, model_path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss
    }, model_path)
    print(f"Model saved to {model_path}.")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    set_random_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载图数据
    with open('flattened_fc_graph.pkl', 'rb') as f:
        dataset = pickle.load(f)

    if isinstance(dataset, list):
        graph = dataset[0]
    else:
        graph = dataset

    # 获取节点特征
    in_feats = graph.ndata['feature'].shape[1]
    features = graph.ndata['feature']

    # 创建训练和验证掩码
    num_nodes = graph.num_nodes()
    all_indices = np.arange(num_nodes)

    # 划分训练集和验证集（例如，80% 训练，20% 验证）
    train_indices, val_indices = train_test_split(all_indices, test_size=0.2, random_state=42)

    # 创建掩码张量
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_indices] = True

    # 将掩码添加到图的节点数据中
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask

    # 初始化模型
    hidden_feats = 256
    model = GCN(in_feats=in_feats, hidden_feats=hidden_feats, num_classes=in_feats).to(device)
    model_path = 'gae_model_flattened_fc_1021.pth'

    # 初始化优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, min_lr=1e-4)

    # 检查是否存在预训练模型
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            starting_epoch = checkpoint['epoch'] + 1  # 从下一个 epoch 开始
            best_loss = checkpoint['best_loss']
            print(f"Loaded pre-trained model weights from {model_path}. Continuing training from epoch {starting_epoch}.")
        except Exception as e:
            print(f"Error loading model weights: {e}. Starting from scratch.")
            starting_epoch = 1
            best_loss = float('inf')
    else:
        print(f"No pre-trained model found at {model_path}. Starting from scratch.")
        starting_epoch = 1
        best_loss = float('inf')

    # 训练参数
    num_epochs = 1000
    patience = 10
    min_delta = 0.01
    counter = 0

    # 开始训练
    for epoch in range(starting_epoch, num_epochs + 1):
        # 训练一个 epoch
        train_loss = train_epoch(model, optimizer, device, graph)
        val_loss = evaluate(model, graph, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            save_model(model, optimizer, epoch, best_loss, model_path)
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print("Training complete.")
