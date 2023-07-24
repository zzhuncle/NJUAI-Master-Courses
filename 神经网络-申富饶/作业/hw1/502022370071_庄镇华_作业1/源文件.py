import numpy as np
from matplotlib import pyplot as plt
global_lr = 100 # 学习率调节

def moon(N, w, r, d):
    '''
    # :param w: 半月宽度 
    # :param r: x 轴偏移量 
    # :param d: y 轴偏移量
    # :param N: 半月散点数量 
    # :return: data (2*N*3) 月亮数据集 data_dn (2*N*1) 标签 
    '''
    data = np.ones((2*N,4))
    # 半月 1 的初始化
    r1 = 10 # 半月 1 的半径,圆心
    np.random.seed(1919810)
    w1 = np.random.uniform(-w / 2, w / 2, size=N) # 半月 1 的宽度范围
    theta1 = np.random.uniform(0, np.pi, size=N) # 半月 1 的角度范围
    x1 = (r1 + w1) * np.cos(theta1) # 行向量
    y1 = (r1 + w1) * np.sin(theta1)
    label1 = [1 for i in range(1,N+1)] # label for Class 1
    # 半月 2 的初始化
    r2 = 10 # 半月 2 的半径,圆心
    w2 = np.random.uniform(-w / 2, w / 2, size=N) # 半月 2 的宽度范围
    theta2 = np.random.uniform(np.pi, 2 * np.pi, size=N) # 半月 2 的角度范围
    x2 = (r2 + w2) * np.cos(theta2) + r
    y2 = (r2 + w2) * np.sin(theta2) - d
    label2 = [-1 for i in range(1,N+1)] # label for Class 2
    data[:,1] = np.concatenate([x1, x2])
    data[:,2] = np.concatenate([y1, y2])
    data[:,3] = np.concatenate([label1, label2])
    return x1, y1, x2, y2, data

losses = []
class Perceptron(object):
    def __init__(self, x, y, learning_rate):
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.w = np.zeros(x.shape[1]) # 权重
        self.b = 0 # 偏置
        self.activate_func = np.sign # 激活函数
        self.out = None # 权重
        
    def calculate(self, x):
        return self.activate_func(np.dot(self.w, x.T) + self.b)
    
    def update(self, x, y):
        self.w += self.learning_rate * x.T * (y - self.out)
        self.b += self.learning_rate * (y - self.out)
    
    def train(self, epochs):
        for _ in range(epochs):
            loss = 0
            for i in range(self.x.shape[0]):
                self.out = self.calculate(self.x[i])
                loss += (self.out - self.y[i]) ** 2
                self.update(self.x[i], self.y[i])
            losses.append(loss / self.x.shape[0])
            
x1, y1, x2, y2, data = moon(2000, 6, 10, 2)
x, y = data[:,1:3], data[:,3]
print(x.shape, y.shape)
model = Perceptron(x, y, global_lr)

# 可视化数据集
def plot(x1, y1, x2, y2):
    plt.figure(figsize=(15, 12))
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.rcParams.update({"font.size":20})
    plt.title('bi-month dataset')
    
plot(x1, y1, x2, y2)

# 可视化学习曲线
model.train(100)
plt.figure(figsize=(15, 12))
plt.plot(losses)
plt.xlabel('epoch', fontsize=20)
plt.ylabel('MSE', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# 可视化决策边界
def plot(x1, y1, x2, y2, w, b):
    plt.figure(figsize=(15, 12))
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    x = np.linspace(-15, 25, 50)
    y = -w[0] / w[1] * x - b / w[1]
    plt.plot(x, y, c='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('bi-month dataset')
    
plot(x1, y1, x2, y2, model.w, model.b)

# 打印决策边界参数
print(-model.w[0] / model.w[1], -model.b / -model.w[1])