import numpy as np

def swish(x):
    beta = 0.2
    return x * (1 / (1 + np.exp(-beta * x)))

def swish_derivative(x):
    beta = 0.2
    sigma = (1 / (1 + np.exp(-beta * x)))
    fx = x * sigma
    return beta * fx + sigma * (1 - beta * fx)

class Layer():
    def __init__(self,input_dim,output_dim,bias=True):
        super(Layer, self).__init__()
        # 输入维度
        self.input_dim = input_dim
        # 输出维度
        self.output_dim = output_dim
        # 随机初始化权重
        self.w = np.random.randn(self.input_dim, self.output_dim)*0.1
        self.bias = bias
        # 初始化激活函数
        self.activation = swish
        self.activation_derivative = swish_derivative
        # 随机初始化偏置
        if self.bias:
            self.b = np.random.randn(self.output_dim)*0.1
        
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
    
# 多层感知机
class MultiLayer():
    def __init__(self,layer_dim_list,lr,bias=True):
        super(MultiLayer, self).__init__()
        # 单层感知机列表
        self.layer_list = []
        # 学习率
        self.lr = lr
        for i in range(len(layer_dim_list)-1):
            input_dim = layer_dim_list[i]
            output_dim = layer_dim_list[i+1]
            self.layer_list.append(Layer(input_dim,output_dim,bias))
        # 多层感知机层数
        self.layer_num = len(self.layer_list)
        
    def __call__(self,x,train=False):
        # __call__函数使得类对象具有类似函数的功能
        # 直接内部调用forward函数即可
        z = self.forward(x,train)
        return z
        
    def forward(self,x,train):
        x = x.reshape(x.shape[0], -1)
        out = x
        for layer in self.layer_list:
            out = layer.forward(x,True)
            x = out
        return out
           
    # 反向传播函数
    def backward(self,error,eta):
        # 从后往前进行梯度更新
        for idx in range(self.layer_num-1,-1,-1):
            error = self.layer_list[idx].backward(error,eta)
        return None

if __name__ == '__main__':
    x = np.linspace(-0.5, 0.6, 1000, endpoint=True)
    y = np.cos(2*np.pi*x)
    x_train, y_train = x.reshape(x.shape[0], 1), y.reshape(y.shape[0], 1)
    
    np.random.seed(2028)
    layer = MultiLayer([1,8,1], 0.00005, True)
    print(layer.layer_num)

    for i in range(80000):           
        y_pred = layer.forward(x_train, True) 
        error = y_pred - y_train
        layer.backward(error, 0.0004)    
        if i % 800 == 0:                
            mse = np.mean(np.square(error))                 
            print(mse)