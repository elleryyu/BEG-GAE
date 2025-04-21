import numpy as np
import pandas as pd
import torch
import os
from torch import nn
from sklearn.preprocessing import StandardScaler
from model import MLP  # 假设您已定义 MLP 模型
from utils import set_random_seed

# 1. 设置随机种子
set_random_seed(42)

# 2. 选择设备（GPU 或 CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. 加载数据
data = pd.read_csv('../site_16/fc_and_labels_transformed.csv', index_col=0)

# 定义标签列
label_cols = ['Anx', 'OCD', 'ADHD', 'ODD', 'Cond']
label_cols = [col for col in label_cols if col in data.columns]

# 提取特征和标签
fc_cols = data.columns.difference(label_cols)
X = data[fc_cols].values
y = data[label_cols].values

# 将特征数据转换为 Tensor 并移到设备
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

# 设置输入维度和隐藏层维度
input_dim = X.shape[1]  # 71631
print(f"Input dimension: {input_dim}")
hidden_dim = 1024  # 设置隐藏层维度
latent_dim = 256   # 设置潜在层维度


model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 设置训练参数
num_epochs = 500
best_loss = float('inf')
patience = 5
counter = 0

# 创建模型保存目录
model_dir = 'saved_models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)  # 如果目录不存在则创建
model_path = os.path.join(model_dir, 'mlp_best.pth')

# 检查是否存在已保存的模型
if os.path.exists(model_path):
    print(f"Model found at {model_path}, loading model and continuing training...")
    model.load_state_dict(torch.load(model_path))  # 加载已保存的模型权重
else:
    print("Model path not found, starting training from scratch...")

# 5. 开始训练循环
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式

    # 前向传播
    decoded, encoded = model(X_tensor)

    # 计算重构损失
    loss = criterion(decoded, X_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印每个 epoch 的损失
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Early stopping 逻辑
    if loss.item() < best_loss:
        best_loss = loss.item()
        counter = 0
        # 保存最优模型
        torch.save(model.state_dict(), model_path)
        print(f'Validation loss decreased, saving model to {model_path}')
    else:
        counter += 1
        if counter >= patience:
            print('Early stopping triggered')
            break

print("Training complete.")
