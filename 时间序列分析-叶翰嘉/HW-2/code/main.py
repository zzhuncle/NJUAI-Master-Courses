import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from models import MLForecastModel, ZeroForecast, ARForecast, EMAForecast
from transforms import Transform, IdentityTransform, NormalizationTransform, StandardizationTransform, MeanNormalizationTransform, BoxCoxTransform
from metrics import *

models_map = {'AR' : ARForecast(), 'EMA(0.2)' : EMAForecast(0.2), 'EMA(0.5)' : EMAForecast(0.5), 'EMA(0.8)' : EMAForecast(0.8)}
trans_map = {'None' : IdentityTransform(), 'Normalize' : NormalizationTransform(), 'Standardize' : StandardizationTransform(),
             'MeanNormalize' : MeanNormalizationTransform(), 'Box-Cox' : BoxCoxTransform()}

def train(train_X: np.ndarray, train_Y: np.ndarray, transform: Transform, model: MLForecastModel):
    t_X = transform.transform(train_X)
    t_Y = transform.transform(train_Y)
    model.fit(t_X, t_Y)
    return model

def test(test_X: np.ndarray, test_Y: np.ndarray, transform: Transform, model: MLForecastModel):
    test_X = transform.transform(test_X)
    fore = model.forecast(test_X)
    # 将预测做逆变换
    fore = transform.inverse_transform(fore)
    # 计算各个指标上的性能
    losses = { }
    losses['MSE'] = mse(test_Y, fore)
    losses['MAE'] = mae(test_Y, fore)
    losses['MAPE'] = mape(test_Y, fore)
    losses['sMAPE'] = smape(test_Y, fore)
    losses['MASE'] = mase(test_Y, fore)
    return fore, losses

def create_sub_series(series: np.ndarray, window_len: int, horizon: int):
    subseries = sliding_window_view(series, window_len + horizon)
    return subseries[:, :window_len], subseries[:, window_len:]

def main():
    # 读取ETTh1数据
    ETTh1_data = pd.read_csv('../data/ETTh1.csv')
    ETTh1_data.set_index('date', inplace=True)
    ETTh1_data.index = pd.DatetimeIndex(ETTh1_data.index, freq='infer')
    OT_data = ETTh1_data['OT']
    OT_data.plot()
    # print(OT_data)
    # 划分训练集和测试集
    split = 16 * 30 * 24
    train_OT_data, test_OT_data = OT_data[:split], OT_data[split:]

    # 从长序列中构造子序列集合, L历史序列长度,H预测序列长度
    L, H = 96, 32
    train_X, train_Y = create_sub_series(train_OT_data.values, L, H)
    test_X, test_Y = create_sub_series(test_OT_data.values, L, H)
    # print(train_X.shape, train_Y.shape) # (11393, 96) (11393, 32)

    # print(test_X.shape) # (5773, 96)
    df = pd.DataFrame()
    for model_name, model in models_map.items(): # TODO : 替换成具体模型
        for trans_name, trans in trans_map.items(): # TODO : 替换成具体变换
            # 训练模型
            # print(model, trans)
            fitted_model = train(train_X, train_Y, trans, model)
            # 测试模型并获得预测
            forecast, losses = test(test_X, test_Y, trans, fitted_model)
            df = pd.concat([df, pd.DataFrame({'Model' : model_name,
                        'Transform' : trans_name,
                        'MAE' : losses['MAE'],
                        'MSE' : losses['MSE'],
                        'MAPE' : losses['MAPE'],
                        'sMAPE' : losses['sMAPE'],
                        'MASE' : losses['MASE']}, index = [0])], ignore_index = True)
    df.to_csv('result.csv')

if __name__ == '__main__':
    main()
