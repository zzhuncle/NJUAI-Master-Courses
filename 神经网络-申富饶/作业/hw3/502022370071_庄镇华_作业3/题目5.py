import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    grad = np.array(x, copy=True)             
    grad[x >  0] = 1.             
    grad[x <= 0] = 0.             
    return grad

class Layer():
    def __init__(self,input_dim,output_dim,bias=True):
        super(Layer, self).__init__()
        # 输入维度
        self.input_dim = input_dim
        # 输出维度
        self.output_dim = output_dim
        # 随机初始化权重
        self.w = np.random.randn(self.input_dim, self.output_dim)
        self.bias = bias
        # 初始化激活函数
        self.activation = relu
        self.activation_derivative = relu_derivative
        # 随机初始化偏置
        if self.bias:
            self.b = np.random.randn(self.output_dim)
        
    def __call__(self,x,train=False):
        # __call__函数使得类对象具有类似函数的功能
        # 直接内部调用forward函数即可
        z = self.forward(x,train)
        return z
        
    def forward(self,x,train):
        # 计算wx + b
        y = np.dot(x, self.w) + self.b
        # 计算\sigma(wx + b)
        z = self.activation(y)
        # 如果是训练模式，则需要计算梯度
        if train:
            self.pre_output = x
            self.grad_z = self.activation_derivative(y)
        return z
        
    # 反向传播函数
    def backward(self,error,eta):
        '''
        反向传播过程 error->out->net->w,b
        '''
        # d_out_net代表out对net的求导结果
        d_out_net = self.grad_z 
        # d_net_w代表net对w的求导结果
        d_net_w = self.pre_output
        
        
        # d_error_net代表error对net的求导结果，这里应用了链式法则
        d_error_net = np.multiply(d_out_net, error)

        # d_error_w代表error对w的求导结果，这里应用了链式法则
        d_error_w = np.dot(d_net_w.T, d_error_net)
        
        self.dw = d_error_w
        
        if self.bias:
            d_error_b = d_error_net.sum(axis=0)
            self.db = d_error_b
            
        # 给下一层的error对out的求导的结果为上一层的加权和
        self.d_error_out_ = np.dot(d_error_net, self.w.T)
        
        # eta为学习率，进行梯度更新
        self.w -= eta * self.dw
        self.b -= eta * self.db
        
        return self.d_error_out_
    
if __name__ == '__main__':
    x, y = datasets.make_moons(n_samples=1000, noise=0.2, random_state=100)
    x, y = x.reshape(1000, 2), y.reshape(1000, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    print(x.shape, y.shape)

    np.random.seed(2023)
    layer = Layer(2, 1, True)

    for i in range(800):           
        y_pred = layer.forward(x_train, True) 
        error = y_pred - y_train
        layer.backward(error, 0.0001)             
        if i % 80 == 0:                
            mse = np.mean(np.square(error))                 
            print(mse)