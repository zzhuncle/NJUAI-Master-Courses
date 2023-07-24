import numpy as np

class ReLU():
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, dz): # dz为梯度
        dx = dz * np.where(self.mask, 1, 0)
        return dx

class LeakyReLU():
    def __init__(self, alpha = 0.02):
        self.alpha = alpha
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        self.mask = x > 0
        return np.where(self.mask, x, self.alpha * x)
    
    def backward(self, dz):
        return dz * np.where(self.mask, 1, self.alpha)
    
class Sigmoid():
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        x = np.clip(x, -500, 500) # 防止np.exe(-x)提出
        self.sigmoid_x = 1 / (1 + np.exp(-x))
        return self.sigmoid_x

    def backward(self, dz):
        dx = dz * self.sigmoid_x * (1 - self.sigmoid_x)
        return dx

class Softmax():
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        probs = np.clip(probs, 1e-10, None) # 防止溢出
        return probs
    
    def backward(self, probs, labels):
        return probs - labels

class Linear():
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # 初始化权重和偏置
        self.w = np.sqrt(2.0 / input_dim) * np.random.randn(output_dim, input_dim)
        self.b = np.zeros((1, output_dim))
        # 保存中间结果
        self.x = None
        # 保存权重和偏置的梯度
        self.dw = None
        self.db = None
        
        self.parameters = [
            {'value': self.w, 'grad': self.dw},
            {'value': self.b, 'grad': self.db}
        ]
        
    def forward(self, x):
        # 前向传播
        self.x = x
        return np.dot(x, self.w.T) + self.b
    
    def __call__(self, x):
        return self.forward(x)
    
    def backward(self, dz):
        # 后向传播
        self.dw = np.dot(dz.T, self.x) 
        self.db = np.sum(dz, axis=0, keepdims=True)
        
        self.parameters[0]['grad'] = self.dw
        self.parameters[1]['grad'] = self.db
        
        dx = np.dot(dz, self.w)
        return dx

class MLP():
    def __init__(self, input_dim, output_dim, layers):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = layers
        self.parameters = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                self.parameters += layer.parameters
    
    def __call__(self, x):
        return self.forward(x)
                
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, probs, labels):
        # 反向传播
        # softmax梯度单独处理
        d_out_net = self.layers[-1].backward(probs, labels)
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[layer_idx]
            d_out_net = layer.backward(d_out_net)
