import numpy as np

# 衰减学习率调度器
class LearningRateScheduler():
    def __init__(self, optimizer, patience, factor=0.5, min_learning_rate=1e-6):
        self.optimizer = optimizer
        self.learning_rate = self.optimizer.learning_rate
        self.patience = patience # 容忍迭代次数
        self.factor = factor # 衰减因子
        self.min_learning_rate = min_learning_rate
        self.counter = 0
        self.best_loss = None

    def step(self, current_loss):
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.learning_rate = max(self.learning_rate * self.factor, self.min_learning_rate)
            self.counter = 0
        self.optimizer.learning_rate = self.learning_rate

# 随机梯度下降优化器
class SGD():
    def __init__(self, params, learning_rate=0.001, weight_decay=0, regularization=None):
        self.params = params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.regularization = regularization # 是否采用正则化

    def step(self):
        for param in self.params:
            if self.regularization == 'l1':
                param['value'] -= self.learning_rate * (param['grad'] + self.weight_decay * np.sign(param['value']))
            elif self.regularization == 'l2':
                param['value'] -= self.learning_rate * (param['grad'] + self.weight_decay * param['value'])
            else:
                param['value'] -= self.learning_rate * param['grad']

    def zero_grad(self):
        for param in self.params:
            param['grad'] = np.zeros_like(param['value'])

# Adam优化器
class Adam():
    def __init__(self, params, learning_rate=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, regularization=None):
        self.params = params
        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps # 防止溢出错误
        self.weight_decay = weight_decay
        self.regularization = regularization
        self.m = {}
        self.v = {}
        for i, param in enumerate(self.params):
            self.m[i] = np.zeros_like(param['value']) # 一阶矩估计
            self.v[i] = np.zeros_like(param['value']) # 二阶矩估计

    def step(self):
        for i, param in enumerate(self.params):
            if self.regularization == 'l1':
                param['grad'] += self.weight_decay * np.sign(param['value'])
            elif self.regularization == 'l2':
                param['grad'] += self.weight_decay * param['value']

            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param['grad']
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * param['grad'] ** 2

            m_hat = self.m[i] / (1 - self.betas[0])
            v_hat = self.v[i] / (1 - self.betas[1])

            param['value'] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            
    def zero_grad(self):
        for param in self.params:
            param['grad'] = np.zeros_like(param['value'])
