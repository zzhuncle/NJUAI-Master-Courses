import numpy as np

class MLForecastModel:
    def __init__(self) -> None:
        super().__init__()
        self.fitted = False
        self.horizon = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        :param X: 用于训练的历史序列集合
        :param Y: 历史序列对应的未来序列集合
        """
        self.horizon = Y.shape[1]
        self._fit(X, Y)
        self.fitted = True

    def _fit(self, X: np.ndarray, Y: np.ndarray):
        raise NotImplementedError

    def _forecast(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forecast(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: 输入的历史序列
        :return: 预测的未来序列值
        """
        if not self.fitted:
            raise ValueError("模型未训练")
        pred = self._forecast(X)
        return pred


class ZeroForecast(MLForecastModel):
    def _fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], self.horizon))


# 线性回归模型
class ARForecast(MLForecastModel):
    # 使用最小二乘估计闭式解
    def _fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        X = np.pad(X, ((0, 0), (0, 1)), 'constant', constant_values = (0, 1)) # 添加非齐次项 多维数组 np.pad 函数
        X, Y = np.mat(X), np.mat(Y)
        self.theta = (X.T @ X).I @ X.T @ Y # [97, 32]

    def _forecast(self, X: np.ndarray) -> np.ndarray:
        X = np.pad(X, ((0, 0), (0, 1)), 'constant', constant_values = (0, 1))
        return np.array(np.mat(X) @ self.theta)


# 指数平滑模型
class EMAForecast(MLForecastModel):
    def __init__(self, alpha = 0.2) -> None: # 对于变化缓慢的序列，常取较小值；相反，对于变化迅速的序列，常取较大值
        super().__init__()
        self.alpha = alpha
        self.lamda = 1 - self.alpha
        self.fitted = False
        self.horizon = None

    def _fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass

    def _forecast(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[-1]
        c = (1 - self.lamda)
        lamda_list = np.array([self.lamda ** (N - i - 1) for i in range (N)])
        return c * (lamda_list * X).sum(axis = 1).reshape(-1, 1).repeat(self.horizon, axis = 1)
