from utils import get_intersection_values
import numpy as np


def mse(target, predict):
    # print(target.index, predict.index)
    target_values, predict_values = get_intersection_values(target, predict)
    return np.mean((target_values - predict_values) ** 2)


def mae(target, predict):
    target_values, predict_values = get_intersection_values(target, predict)
    return np.mean(np.abs(target_values - predict_values))

'''
def mape(target, predict):
    target_values, predict_values = get_intersection_values(target, predict)
    return np.mean(np.abs((target_values - predict_values) / target_values)) * 100
'''

def mape(target, predict):
    target_values, predict_values = get_intersection_values(target, predict)
    # 除以零特例
    idxes = (target_values == 0)
    target_values[idxes] = predict_values[idxes]
    N = len(target_values) - np.sum(idxes)
    return np.sum(np.abs((target_values - predict_values) / target_values)) / N * 100

def smape(target, predict):
    target_values, predict_values = get_intersection_values(target, predict)
    return np.mean(np.abs(target_values - predict_values) / (np.abs(target_values) + np.abs(predict_values))) * 200


def mase(target, predict, T = 24):
    target_values, predict_values = get_intersection_values(target, predict)
    scale, N = 0, len(target_values)
    for i in range(T, N):
        scale += np.abs(target_values[i] - target_values[i - T])
    scale /= (N - T)
    return np.mean(np.abs(target_values - predict_values) / scale)
