# %%
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.unicode_minus'] = False    #显示负号
plt.rc('font',family='Times New Roman')

# 加载数据集
boston = load_boston()
X = boston.data
y = boston.target

# 以7：3分割训练集和数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# 对数据进行标准归一化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 转化为tensor张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.float)
X_val_tensor = torch.tensor(X_val, dtype=torch.float)
y_val_tensor = torch.tensor(y_val, dtype=torch.float)

X_train_scaled_tensor = torch.tensor(X_train_scaled, dtype=torch.float)
X_val_scaled_tensor = torch.tensor(X_val_scaled, dtype=torch.float)

# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


# 训练过程
def train(model, X_train, y_train, X_val, y_val, num_epochs=5):
    # SGD优化器
    loss_fn = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.001)
    train_losses = []
    val_losses = []
    val_pred_list = []

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train() # 模型训练
        for xb, yb in train_dl:
            y_pred = model(xb)
            loss = loss_fn(y_pred.squeeze(), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_losses.append(loss.item())
        model.eval() # 模型评估
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = loss_fn(y_val_pred.squeeze(), y_val)
            val_losses.append(val_loss.item())
            val_pred_list.append(y_val_pred)
        print(f'Epoch {epoch+1}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}')
    return train_losses, val_losses, val_pred_list

# 不对数据进行归一化
model = MLP()
train_losses, val_losses, val_pred_list = train(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

# 对数据进行归一化
model_scaled = MLP()
train_losses_scaled, val_losses_scaled, val_pred_list_scaled = train(model_scaled, X_train_scaled_tensor, y_train_tensor, X_val_scaled_tensor, y_val_tensor)

# 绘制训练损失
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
# 使用双y轴让差异更清晰
axes0_twin = axes[0].twinx() 
axes[0].plot(range(1,6),train_losses, label='Without Normalization', color='m')
axes0_twin.plot(range(1,6),train_losses_scaled, label='With Normalization', color='c')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Train Loss Without Normalization', color='m')
axes0_twin.set_ylabel('Train Loss With Normalization', color='c')
axes[0].set_title('Training Losses')
lines, labels = axes[0].get_legend_handles_labels()
lines2, labels2 = axes0_twin.get_legend_handles_labels()
axes[0].legend(lines + lines2, labels + labels2, loc=0)

# 绘制真实值和预测值差异
mean_difference_without_norm = [np.abs(np.mean(y_val_tensor.detach().numpy() - pred.detach().numpy())) for pred in val_pred_list]
mean_difference_with_norm = [np.abs(np.mean(y_val_tensor.detach().numpy() - pred.detach().numpy())) for pred in val_pred_list_scaled]
# 使用双y轴使差异更清晰
axes1_twin = axes[1].twinx()
axes[1].plot(range(1,6),mean_difference_without_norm, label='Without Normalization', color='m')
axes1_twin.plot(range(1,6),mean_difference_with_norm, label='With Normalization', color='c')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Mean Difference Without Normalization', color='m')
axes1_twin.set_ylabel('Mean Difference With Normalization', color='c')
axes[1].set_title('Mean Difference between True and Predicted Values')
lines, labels = axes[1].get_legend_handles_labels()
lines2, labels2 = axes1_twin.get_legend_handles_labels()
axes[1].legend(lines + lines2, labels + labels2, loc=0)

plt.tight_layout()
plt.savefig('./boston1.png', dpi=400)

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import SGD
from torch.nn import MSELoss, L1Loss, SmoothL1Loss  # 导入不同的损失函数
num_epochs = 1000

# 修改训练函数以接受损失函数作为参数
def train(model, X_train, y_train, X_val, y_val, loss_fn, num_epochs=1000):
    optimizer = SGD(model.parameters(), lr=0.001)
    train_losses = []
    val_losses = []
    val_pred_list = []

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    
    for epoch in range(num_epochs):
        model.train() # 模型训练
        for xb, yb in train_dl:
            y_pred = model(xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss = loss_fn(y_pred.squeeze(), yb)
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = loss_fn(y_val_pred.squeeze(), y_val)
            val_losses.append(val_loss.item())
            val_pred_list.append(y_val_pred)

        print(f'Epoch {epoch+1}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

    return train_losses, val_losses, val_pred_list

# 定义不同的损失函数
loss_functions = [ L1Loss(), MSELoss(),SmoothL1Loss()]  # 平方误差损失，绝对值误差损失和 Huber 损失
loss_names = [ 'L1', 'MSE','Huber']

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# 对于每个损失函数，训练模型并绘制结果
for loss_fn, loss_name in zip(loss_functions, loss_names):
    model = MLP()
    train_losses, val_losses, val_pred_list = train(model, X_train_scaled_tensor, y_train_tensor, X_val_scaled_tensor, y_val_tensor, loss_fn)

    # 绘制训练损失
    axes[0].plot(range(1,num_epochs+1), train_losses, label=f'{loss_name}')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Losses')
    axes[0].legend()

    # 计算并绘制预测与真实值的平均差值
    mean_difference = [np.mean(y_val_tensor.detach().numpy() - pred.detach().numpy()) for pred in val_pred_list]
    axes[1].plot(range(1,num_epochs+1), mean_difference, label=f'{loss_name}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Difference')
    axes[1].set_title('Mean Difference between True and Predicted Values')
    axes[1].legend()

plt.tight_layout()
plt.savefig('./boston2.png', dpi=400)

# %%
from torch.optim import SGD, lr_scheduler  # 导入学习率衰减函数


learning_rates = [1e-4, 1e-3, 0.01]  # 三种不同的学习率
lr_decay_factor = 0.1  # 学习率衰减因子
decay_steps = 3  # 每隔多少个epoch衰减一次学习率

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
loss_fn = MSELoss()  # 使用MSE作为损失函数

def train(model, X_train, y_train, X_val, y_val, loss_fn, optimizer, num_epochs=1000):
    train_losses = []
    val_losses = []
    val_pred_list = []

    for epoch in range(num_epochs):
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred.squeeze(), y_train)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = loss_fn(y_val_pred.squeeze(), y_val)
            val_losses.append(val_loss.item())
            val_pred_list.append(y_val_pred)

        print(f'Epoch {epoch+1}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

    return train_losses, val_losses, val_pred_list
plt.rcParams['axes.unicode_minus'] = False    #显示负号
plt.rc('font',family='Times New Roman')
# plt.rcParams['font.sans-serif'] = ['SimHei'] # plt输出中文
# 对于每个学习率，训练模型并绘制结果
for lr in learning_rates:
    model = MLP()
    optimizer = SGD(model.parameters(), lr=lr)
    train_losses, val_losses, val_pred_list = train(model, X_train_scaled_tensor, y_train_tensor, X_val_scaled_tensor, y_val_tensor, loss_fn, optimizer, num_epochs=100)

    axes0_0 = axes[0]
    axes0_1 = axes[1]
    axes0_0.plot(train_losses, label=f'lr={lr}')
    axes0_0.set_xlabel('Epoch')
    axes0_0.set_ylabel('Train Loss')
    axes0_0.set_title('Training Losses')
    axes0_0.legend()

    # 计算并绘制预测与真实值的平均差值
    mean_difference = [np.abs(np.mean(y_val_tensor.detach().numpy() - pred.detach().numpy())) for pred in val_pred_list]
    axes0_1.plot(mean_difference, label=f'lr={lr}')
    axes0_1.set_xlabel('Epoch')
    axes0_1.set_ylabel('Mean Difference')
    axes0_1.set_title('Mean Difference between True and Predicted Values')
    axes0_1.legend()
plt.savefig('./boston3.png', dpi=500)

# %%
import matplotlib.pyplot as plt


from torch.optim import SGD, lr_scheduler
lr = 1e-2 # 学习率
num_epochs = 100 # 总运行次数
lr_decay_factor = 0.8  # 学习率衰减因子
decay_steps = 3  # 每隔多少个epoch衰减一次学习率

def train_with_scheduler(model, X_train, y_train, X_val, y_val, loss_fn, num_epochs=1000, use_scheduler=False):
    optimizer = SGD(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_steps, gamma=lr_decay_factor)  # 定义学习率衰减策略
    train_losses = []
    val_losses = []
    val_pred_list = []

    for epoch in range(num_epochs):
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred.squeeze(), y_train)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() # 更新学习率
        
        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val)
            val_loss = loss_fn(y_val_pred.squeeze(), y_val)
            val_losses.append(val_loss.item())
            val_pred_list.append(y_val_pred)
        print(f'Epoch {epoch+1}, Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}')
    return train_losses, val_losses, val_pred_list

model = MLP()
# 使用学习率衰减策略
train_losses_schedule, val_losses_schedule, val_pred_list_schedule = train_with_scheduler(model, X_train_scaled_tensor, y_train_tensor, 
                                                               X_val_scaled_tensor, y_val_tensor, loss_fn, num_epochs, True)
model = MLP()
# 不适用学习率衰减策略
train_losses, val_losses, val_pred_list = train_with_scheduler(model, X_train_scaled_tensor, y_train_tensor, 
                                                               X_val_scaled_tensor, y_val_tensor, loss_fn, num_epochs, False)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

# 使用双y轴让差异更清晰
axes[0].plot(range(1,num_epochs+1),train_losses, label='Without lr Scheduler')
axes[0].plot(range(1,num_epochs+1),train_losses_schedule, label='With lr Scheduler')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Train Loss')
axes[0].set_title('Training Losses')
axes[0].legend()

# 绘制真实值和预测值差异
mean_difference_without_schedule = [np.abs(np.mean(y_val_tensor.detach().numpy() - pred.detach().numpy())) for pred in val_pred_list]
mean_difference_with_schedule = [np.abs(np.mean(y_val_tensor.detach().numpy() - pred.detach().numpy())) for pred in val_pred_list_schedule]
axes[1].plot(range(1,num_epochs+1),mean_difference_without_schedule, label='Without lr Scheduler')
axes[1].plot(range(1,num_epochs+1),mean_difference_with_schedule, label='With lr Scheduler')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Mean Difference')
axes[1].set_title('Mean Difference between True and Predicted Values')
axes[1].legend()

plt.tight_layout()
plt.savefig('./boston4.png', dpi=500)


