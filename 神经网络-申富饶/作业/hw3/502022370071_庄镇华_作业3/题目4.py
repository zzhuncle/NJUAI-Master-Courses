import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def him(x):
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2

x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
X,Y = np.meshgrid(x,y)
Z = him([X,Y])

import torch
lr = 1e-3 # 学习率
T = 20000 # 迭代次数
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X,Y,Z,cmap='rainbow')

if __name__ == '__main__':
    # x代表坐标值(x,y)
    x = torch.tensor([0., 0.], requires_grad=True)
    # 定义Adam优化器，学习速率是1e-3
    optimizer = torch.optim.Adam([x], lr=lr)
    for step in range(T):
        # 输入坐标，得到预测值
        pred = him(x)
        # 梯度清零
        optimizer.zero_grad()
        # 梯度回传，获取坐标的梯度信息
        pred.backward()
        # 沿梯度方向更新梯度，优化坐标值
        optimizer.step()

        if step % (T // 10) == 0:
            print('step {}: x = {}, f(x) = {}'.format(step, x.tolist(), pred.item()))