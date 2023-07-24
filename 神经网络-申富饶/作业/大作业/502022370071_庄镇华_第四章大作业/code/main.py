import gzip
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 以下引入自己实现的Dataloader、SGD、Adam、MLP等
from data import Dataloader
from optimizer import SGD, Adam, LearningRateScheduler
from model import ReLU, LeakyReLU, Sigmoid, Softmax, Linear, MLP

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28 * 28)
    return data / 255.0

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return np.eye(10)[data]

# 交叉熵损失函数
def cross_entropy(y_pred, y_true):
    n_samples = y_true.shape[0]
    loss = -np.sum(np.log(y_pred + 1e-10) * y_true) / n_samples
    return loss

# 计算精度
def accuracy(y_pred, y_true):
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    return np.sum(y_pred == y_true) / len(y_true)
    
dataset_dir = './dataset/'
# 设定随机种子
random.seed(42)
np.random.seed(42)

if __name__ == "__main__":
    input_dim = 28 * 28
    output_dim = 10
    learning_rate = 0.0005
    weight_decay = 0.00001
    regularization = None
    batch_size = 600
    patience = 6
    factor = 0.5
    
    # 定义神经网络层
    layers = [
            Linear(input_dim, 500),
            ReLU(),
            Linear(500, 250),
            LeakyReLU(),
            Linear(250, 50),
            ReLU(),
            Linear(50, output_dim),
            Softmax()
        ]
    # 加载数据
    x_train = load_mnist_images(dataset_dir + 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(dataset_dir + 'train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images(dataset_dir + 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(dataset_dir + 't10k-labels-idx1-ubyte.gz')
    
    # 定义数据记载器、模型、优化器、调度器
    train_loader = Dataloader(x_train, y_train, batch_size)
    test_loader = Dataloader(x_test, y_test, batch_size)
    mlp = MLP(input_dim, output_dim, layers)
    optimizer = Adam(mlp.parameters, learning_rate=learning_rate, regularization=regularization, weight_decay=weight_decay)
    # scheduler = LearningRateScheduler(optimizer, patience, factor)

    num_epochs = 200
    results = []
    # 开始训练
    for epoch in range(num_epochs):
        # 训练阶段
        train_losses = []
        train_accuracies = []
        for batch_images, batch_labels in train_loader:
            outputs = mlp.forward(batch_images)
            train_loss = cross_entropy(outputs, batch_labels)
            train_losses.append(train_loss)
            train_accuracy = accuracy(outputs, batch_labels)
            train_accuracies.append(train_accuracy)
            mlp.backward(outputs, batch_labels)
            optimizer.step()
            optimizer.zero_grad()
        
        # 测试阶段
        test_losses = []
        test_accuracies = []
        for batch_images, batch_labels in test_loader:
            outputs = mlp.forward(batch_images)
            test_loss = cross_entropy(outputs, batch_labels)
            test_losses.append(test_loss)
            test_accuracy = accuracy(outputs, batch_labels)
            test_accuracies.append(test_accuracy)
        
        print('Epoch: {}, Train Loss: {:.4f}, Train Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(epoch+1, np.mean(train_losses), np.mean(train_accuracies), np.mean(test_losses), np.mean(test_accuracies)))
        results.append({
            'Epoch': epoch+1,
            'Train Loss': np.mean(train_losses),
            'Train Accuracy': np.mean(train_accuracies),
            'Test Loss': np.mean(test_losses),
            'Test Accuracy': np.mean(test_accuracies),
        })
        # scheduler.step(np.mean(train_losses))

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv('training_results.csv', index=False)

    # 画图
    df = pd.read_csv('training_results.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
    ax1.plot(df['Epoch'], df['Test Loss'], label='Test Loss')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(df['Epoch'], df['Train Accuracy'], label='Train Accuracy')
    ax2.plot(df['Epoch'], df['Test Accuracy'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.savefig("training_results.png", dpi=600)
    plt.show()
