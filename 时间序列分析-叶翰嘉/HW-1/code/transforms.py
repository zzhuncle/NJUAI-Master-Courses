import numpy as np
import pandas as pd

class Transform:
    """
    Transform类用于对时间序列进行预处理，其需要实现transform(变换)和inverse_transform(逆变换)
    """

    def transform(self, data):
        """
        :param data: 时间序列
        :return: 变换后的时间序列
        """
        raise NotImplementedError

    def inverse_transform(self, data):
        """
        :param data: 时间序列
        :return: 逆变换后的时间序列
        """
        raise NotImplementedError

class IdentityTransform(Transform):
    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

# 归一化
class NormalizationTransform(Transform):
    def transform(self, data):
        self.minv = data.min()
        self.maxv = data.max()
        if self.minv != self.maxv:
            data = (data - self.minv) / (self.maxv - self.minv)
        else:
            data -= self.minv
        return data

    def inverse_transform(self, data):
        data = self.minv + data * (self.maxv - self.minv)
        return data

# 标准化
class StandardizationTransform(Transform):
    def transform(self, data):
        self.meanv = data.mean()
        self.stdv = data.std()
        if self.stdv != 0:
            data = (data - self.meanv) / self.stdv
        else:
            data -= self.meanv
        return data

    def inverse_transform(self, data):
        data = self.meanv + data * self.stdv
        return data

# 均值归一化
class MeanNormalizationTransform(Transform):
    def transform(self, data):
        self.meanv = data.mean()
        self.range = data.max() - data.min()
        if self.range != 0:
            data = (data - self.meanv) / self.range
        else:
            data -= self.meanv
        return data

    def inverse_transform(self, data):
        data = self.meanv + data * self.range
        return data

# 二参数Box-Cox变换（处理含有负数的序列）
class BoxCoxTransform(Transform):
    def __init__(self, lamda = 1, lamda2 = 5): # 最小的负数 >= -5
        self.lamda = lamda
        self.lamda2 = lamda2

    def transform(self, data):
        if self.lamda == 0:
            data = np.log(data + self.lamda2)
        else:
            data = ((data + self.lamda2) ** self.lamda - 1) / self.lamda
        return data

    def inverse_transform(self, data):
        if self.lamda == 0:
            data = np.exp(data) - self.lamda2
        else:
            data = (data * self.lamda + 1) ** (1 / self.lamda) - self.lamda2
        return data
