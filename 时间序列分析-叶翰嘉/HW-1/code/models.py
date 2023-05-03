import numpy as np
import pandas as pd
from utils import build_forecast_series


class ForecastModel:
    def __init__(self) -> None:
        super().__init__()
        self.fitted = False
        self.train_series = None

    def fit(self, X: pd.Series) -> None:
        """
        :param X: 用于训练的时间序列, 长度为t
        """
        self.train_series = X
        self._fit(X)
        self.fitted = True

    def _fit(self, X: pd.Series):
        raise NotImplementedError

    def _forecast(self, horizon: int) -> np.ndarray:
        raise NotImplementedError

    def forecast(self, horizon: int) -> pd.Series:
        """
        :param horizon: 预测序列长度
        :return: 预测的未来序列值，长度为horizon
        """
        if not self.fitted:
            raise ValueError("模型未训练")
        pred = self._forecast(horizon)
        return build_forecast_series(pred, self.train_series)

class ZeroForecast(ForecastModel):
    def _fit(self, X: pd.Series) -> None:
        pass

    def _forecast(self, horizon: int) -> np.ndarray:
        pass

# Naive1
class Naive1Forecast(ForecastModel):
    def _fit(self, X: pd.Series) -> None:
        self.last = X[len(X) - 1]

    def _forecast(self, horizon: int) -> np.ndarray:
        return np.zeros((horizon,)) + self.last

# NaiveS（以24小时为周期）
class NaiveSForecast(ForecastModel):
    def __init__(self, T = 24):
        self.T = T

    def _fit(self, X: pd.Series) -> None:
        self.lastT = X[len(X) - self.T : ]

    def _forecast(self, horizon: int) -> np.ndarray:
        preds = np.zeros((horizon,))
        for i in range(horizon):
            preds[i] += self.lastT[i % self.T]
        return preds

# Drift
class DriftForecast(ForecastModel):
    def _fit(self, X: pd.Series) -> None:
        self.last = X[len(X) - 1]
        self.k = (X[len(X) - 1] - X[0]) / (len(X) - 1)

    def _forecast(self, horizon: int) -> np.ndarray:
        preds = np.zeros((horizon,))
        for i in range(horizon):
            preds[i] += self.last + (i + 1) * self.k
        return preds