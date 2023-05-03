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
    periods = [1, 2, 3, 4 , 5, 6, 9, 12, 24, 36, 48, 60, 72, 120, 168, 336, 720, 1440, 2160]
    model_name_list = ['NaiveS-1', 'NaiveS-2', 'NaiveS-3','NaiveS-4','NaiveS-5' ,'NaiveS-6', 'NaiveS-9', 'NaiveS-12', 'NaiveS-24', 'NaiveS-36', 'NaiveS-48', 'NaiveS-60', 'NaiveS-72', 'NaiveS-120', 'NaiveS-168', 'NaiveS-336', 'NaiveS-720', 'NaiveS-1440', 'NaiveS-2160']
    trans_name_list = ['Normalization',]
    model_list = [NaiveSForecast(t) for t in periods]
    trans_list = [NormalizationTransform()]
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
            df = df.append({'model' : model_name_list[i],
            'trans' : trans_name_list[j],
            'mse' : perform['mse'],
            'mae' : perform['mae'],
            'mape%' : perform['mape'],
            'smape%' : perform['smape'],
            'mase' : perform['mase'],}, ignore_index = True)

    # 绘制预测与真实值的曲线
    for mt in ['mse', 'mae', 'mape%', 'smape%', 'mase']:
        fig = plt.figure(figsize = (9, 6))
        plt.plot(range(len(periods)), list(df[mt]))
        plt.xticks(range(len(periods)), periods)
        plt.xlabel("T/hours")
        plt.ylabel(mt)
        plt.savefig(f'{mt}_period.png', bbox_inches = 'tight')
        plt.close(fig)
    df.to_excel('result2.xlsx')

if __name__ == '__main__':
    main()
