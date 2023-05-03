import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from model import Resnet18
from ema import ModelEMA
use_ema = False

# 记录损失和测试准确率随时间的变化
transform_train = transforms.Compose([
	transforms.RandomCrop(32, padding = 4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载数据集 CIFAR-10 32 x 32
train_dataset = torchvision.datasets.CIFAR10(root = '../data', train = True, download = True, transform = transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 512, shuffle = True, num_workers = 4)
test_dataset = torchvision.datasets.CIFAR10(root = '../data', train = False, download = True, transform = transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 512, shuffle = True, num_workers = 4)

# 评价指标
def accuracy(pred, true):
	pred = pred.argmax(dim = 1)
	acc = accuracy_score(true, pred)
	return acc

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20221103)

# 定义损失
myloss = nn.CrossEntropyLoss()
# 定义设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 定义优化器
model = Resnet18(num_classes = 10).to(device)
ema_model = ModelEMA(device, model, 0.999)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0, amsgrad = False)

def train(model, train_loader, test_loader, optimizer):
	loss_list, acc_list = [], []
	# 开始训练
	for epoch in range(1000):
		loss_sum, acc_sum = .0, .0
		model.train()
		for step, (image, label) in enumerate(train_loader):
			image, label = image.to(device), label.to(device)
			pred = model.forward(image)
			loss = myloss(pred, label)
			loss_sum += loss.item()
			loss.backward()
			optimizer.step()
			if use_ema:
				ema_model.update(model)
			optimizer.zero_grad()
		loss_list.append(loss_sum / (step + 1))
		print(f"epoch: {epoch} train_loss: {loss_sum / (step + 1)}")

		model.eval()
		test_model = ema_model.ema if use_ema else model
		for step, (image, label) in enumerate(test_loader):
			image, label = image.to(device), label.to(device)
			with torch.no_grad():
				pred = test_model.forward(image)
				acc = accuracy(pred.cpu(), label.cpu())
			acc_sum += acc.item()
		acc_list.append(acc_sum / (step + 1))
		print(f"epoch: {epoch} test_acc: {acc_sum / (step + 1)}")
		np.save('loss_list_without', loss_list)
		np.save('acc_list_without', acc_list)

train(model, train_loader, test_loader, optimizer)