import pandas as pd
import numpy as np

from models import ForecastModel, ZeroForecast, Naive1Forecast, NaiveSForecast, DriftForecast
from transforms import Transform, IdentityTransform, NormalizationTransform, StandardizationTransform, MeanNormalizationTransform, BoxCoxTransform
from metrics import *
from matplotlib import pyplot as plt

def train(train_data: pd.Series, transform: Transform, model: ForecastModel):
    t_data = transform.transform(train_data)
    model.fit(t_data)
    return model

def test(test_data: pd.Series, transform: Transform, model: ForecastModel):
    fore = model.forecast(len(test_data))
    # 将预测做逆变换
    fore = transform.inverse_transform(fore)

    perform = {}
    # 计算各个指标上的性能
    perform['mse'] = mse(test_data, fore)
    perform['mae'] = mae(test_data, fore)
    perform['mape'] = mape(test_data, fore)
    perform['smape'] = smape(test_data, fore)
    perform['mase'] = mase(test_data, fore)
    return fore, perform

def main():
    # 读取ETTh1数据
    ETTh1_data = pd.read_csv('../data/ETTh1.csv')
    ETTh1_data.set_index('date', inplace = True)
    ETTh1_data.index = pd.DatetimeIndex(ETTh1_data.index, freq = 'infer')
    OT_data = ETTh1_data['OT']
    # print(OT_data.min())

    # 划分训练集和测试集
    split = 16 * 30 * 24
    train_OT_data, test_OT_data = OT_data[ : split], OT_data[split : ]

    df = pd.DataFrame()
    model_name_list = ['Naive1', 'NaiveS', 'Drift']
    trans_name_list = ['None', 'Normalization', 'Standardization', 'MeanNormalization', 'BoxCox--1', 'BoxCox-0', 'BoxCox-0.5', 'BoxCox-1']
    model_list = [Naive1Forecast(), NaiveSForecast(), DriftForecast()]
    trans_list = [IdentityTransform(), NormalizationTransform(), StandardizationTransform(), MeanNormalizationTransform(), BoxCoxTransform(-1),
                BoxCoxTransform(0), BoxCoxTransform(0.5), BoxCoxTransform(1)]
    for i, model in enumerate(model_list):
        for j, trans in enumerate(trans_list):
            '''
            # TODO : 替换成具体模型
            model = ZeroForecast()
            # TODO : 替换成具体变换
            trans = NormalizationTransform()
            '''
            # 训练模型
            fitted_model = train(train_OT_data, trans, model)
            # 测试模型并获得预测
            forecast, perform = test(test_OT_data, trans, fitted_model)
            print(perform)
            df = df.append({'model' : model_name_list[i],
            'trans' : trans_name_list[j],
            'mse' : perform['mse'],
            'mae' : perform['mae'],
            'mape%' : perform['mape'],
            'smape%' : perform['smape'],
            'mase' : perform['mase'],}, ignore_index = True)
            # 绘制预测与真实值的曲线
            fig = plt.figure()
            train_OT_data.plot(label = 'train')
            test_OT_data.plot(label = 'true')
            forecast.plot(label = 'predict')
            plt.savefig(f'{model_name_list[i]}_{trans_name_list[j]}.png', bbox_inches = 'tight')
            plt.close(fig)
    df.to_excel('result.xlsx')

if __name__ == '__main__':
    main()
