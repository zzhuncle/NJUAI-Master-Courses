# %%
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False    #显示负号
plt.rc('font',family='Times New Roman')

# %%
# 设备加载
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
train_data = torchvision.datasets.MNIST(
    root='./MNIST',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

test_data = torchvision.datasets.MNIST(
    root='./MNIST',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False
)

# 数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=100, shuffle=False)

# %%
class MLP(nn.Module):
    def __init__(self, init_method='normal'):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(784, 2048)
        self.layer2 = nn.Linear(2048, 10)
        self.relu = nn.ReLU()
        self.init_method = init_method

    def forward(self, input):
        out = self.layer1(input)
        out = self.relu(out)
        out = self.layer2(out)
        return out

    def reset_parameters(self):
        if self.init_method == 'normal':
            nn.init.normal_(self.layer1.weight)
            nn.init.normal_(self.layer2.weight)
        elif self.init_method == 'zeros':
            nn.init.zeros_(self.layer1.weight)
            nn.init.zeros_(self.layer2.weight)
        elif self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.layer1.weight)
            nn.init.xavier_uniform_(self.layer2.weight)
        else:
            raise ValueError('Invalid init method!')

# %%
# 定义训练函数
def train_model(model, train_loader, test_loader, device, num_epochs, criterion, optimizer):
    n_total_steps = len(train_loader)
    train_acc_list, train_loss_list, test_acc_list = [], [], []

    for _ in range(num_epochs):
        for images, labels in train_loader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            correct_train = 0
            total_train = 0
            for images, labels in train_loader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
            train_acc_list.append(correct_train/total_train)
            train_loss_list.append(loss.item())

            correct_test = 0
            total_test = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
            test_acc_list.append(correct_test/total_test)

    print('Finished Training')

    return train_acc_list, train_loss_list, test_acc_list

# 以不同初始化方法训练并保存结果
init_methods = ['normal', 'zeros', 'xavier']
results = {}

for init_method in init_methods:
    model = MLP(init_method).to(device)
    model.reset_parameters()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    num_epochs = 10

    train_acc_list, train_loss_list, test_acc_list = train_model(model, train_loader, test_loader, device, num_epochs, criterion, optimizer)

    results[init_method] = {
        'train_acc': train_acc_list,
        'train_loss': train_loss_list,
        'test_acc': test_acc_list
    }

# %%
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 绘制训练集准确率折线图
for init_method in init_methods:
    axs[0, 0].plot(range(num_epochs), results[init_method]['train_acc'], label=init_method)
axs[0, 0].set_title('Training Accuracy')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend()

# 绘制训练集损失折线图
for init_method in init_methods:
    axs[0, 1].plot(range(num_epochs), results[init_method]['train_loss'], label=init_method)
axs[0, 1].set_title('Training Loss')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()

# 绘制测试集准确率折线图
for init_method in init_methods:
    axs[1, 0].plot(range(num_epochs), results[init_method]['test_acc'], label=init_method)
axs[1, 0].set_title('Test Accuracy')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].legend()

# 移除右下角的空子图
fig.delaxes(axs[1,1])

# 优化布局
plt.tight_layout()
plt.savefig('./mnist1.png', dpi=500)


# %%
# 定义模型
class MLP(nn.Module):
    def __init__(self, normalization):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(784, 2048)
        self.layer2 = nn.Linear(2048, 10)
        self.relu = nn.ReLU()

        if normalization == 'batch':
            self.norm1 = nn.BatchNorm1d(2048)
            self.norm2 = nn.BatchNorm1d(10)
        elif normalization == 'layer':
            self.norm1 = nn.LayerNorm(2048)
            self.norm2 = nn.LayerNorm(10)
        elif normalization == 'instance':
            self.norm1 = nn.InstanceNorm1d(2048)
            self.norm2 = nn.InstanceNorm1d(10)
        elif normalization == 'group':
            self.norm1 = nn.GroupNorm(32, 2048)  # 假设使用32个组
            self.norm2 = nn.GroupNorm(1, 10)
        else:
            self.norm1 = None
            self.norm2 = None

    def forward(self, input):
        out = self.layer1(input)
        if self.norm1:
            out = self.norm1(out)
        out = self.relu(out)

        out = self.layer2(out)
        if self.norm2:
            out = self.norm2(out)
        return out
    
# 使用不同归一化方法初始化模型
normalizations = ['batch', 'layer', 'instance', 'group']
results = {}

num_epochs = 10
for norm in normalizations:
    model = MLP(norm).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_acc, train_loss, test_acc = train_model(model, train_loader, test_loader, device, num_epochs, criterion, optimizer)
    results[norm] = {'train_acc': train_acc, 'train_loss': train_loss, 'test_acc': test_acc}

# %%
# 绘制结果
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# 训练精度
for norm in normalizations:
    axs[0, 0].plot(range(num_epochs), results[norm]['train_acc'], label=norm)
axs[0, 0].set_title('Training Accuracy')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].legend()

# 训练损失
for norm in normalizations:
    axs[0, 1].plot(range(num_epochs), results[norm]['train_loss'], label=norm)
axs[0, 1].set_title('Training Loss')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].legend()

# 测试精度
for norm in normalizations:
    axs[1, 0].plot(range(num_epochs), results[norm]['test_acc'], label=norm)
axs[1, 0].set_title('Test Accuracy')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].legend()

fig.delaxes(axs[1, 1]) 
plt.tight_layout()
plt.savefig('./mnist2.png', dpi=500)


