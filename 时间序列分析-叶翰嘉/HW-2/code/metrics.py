from utils import get_intersection_values
import numpy as np

def _mse(target, predict):
    return np.mean((target - predict) ** 2)

def _mae(target, predict):
    return np.mean(np.abs(target - predict))

def _mape(target, predict):
    # 除以零特例
    target = target.copy()
    idxes = (target == 0)
    target[idxes] = predict[idxes]
    N = len(target) - np.sum(idxes)
    return np.sum(np.abs((target - predict) / target)) / N * 100

def _smape(target, predict):
    return np.mean(np.abs(target - predict) / (np.abs(target) + np.abs(predict))) * 200

def _mase(target, predict, T = 24):
    scale, N = 0, len(target)
    for i in range(T, N):
        scale += np.abs(target[i] - target[i - T])
    scale /= (N - T)
    return np.mean(np.abs(target - predict) / scale)

def mse(target, predict):
    N = target.shape[0]
    return np.mean([_mse(target[i], predict[i]) for i in range(N)])

def mae(target, predict):
    N = target.shape[0]
    return np.mean([_mae(target[i], predict[i]) for i in range(N)])

def mape(target, predict):
    N = target.shape[0]
    return np.mean([_mape(target[i], predict[i]) for i in range(N)])

def smape(target, predict):
    N = target.shape[0]
    return np.mean([_smape(target[i], predict[i]) for i in range(N)])

def mase(target, predict, T = 24):
    N = target.shape[0]
    return np.mean([_mase(target[i], predict[i]) for i in range(N)])
