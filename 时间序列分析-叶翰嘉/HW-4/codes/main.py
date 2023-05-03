import pandas as pd
import darts
from darts.models import forecasting
# 读取数据
df = pd.read_csv("aus_airpassengers.csv", index_col=0)
series = darts.TimeSeries.from_dataframe(df, "Year", "Passengers")

# 不同模型预测比较
arima = forecasting.auto_arima.AutoARIMA()
model = arima.fit(series = series)
forecast = model.predict(10)
series.plot(label="actual")
forecast.plot(label="AutoARIMA forecast")

arimac = forecasting.arima.ARIMA(p=0, d=1, q=0, trend='t')
modelc = arimac.fit(series = series)
forecastc = modelc.predict(10)
forecastc.plot(label="ARIMA(0, 1, 0) forecast")

arimad = forecasting.arima.ARIMA(p=2, d=1, q=2, trend='t')
modeld = arimad.fit(series = series)
forecastd = modeld.predict(10)
forecastd.plot(label="ARIMA(2, 1, 2) with trend forecast")

arimadd = forecasting.arima.ARIMA(p=2, d=1, q=2, trend=None)
modeldd = arimadd.fit(series = series)
forecastdd = modeldd.predict(10)
forecastdd.plot(label="ARIMA(2, 1, 2) without trend forecast")

# 查看模型参数
modelc.model.summary()

# 查看残差
from matplotlib import pyplot as plt
res = model.model.model_.resid()
plt.xlabel('Year')
plt.ylabel('resid')
plt.plot(df['Year'], res)

# 判断是否属于白噪声
from statsmodels.stats.diagnostic import acorr_ljungbox
ljungbox_result = acorr_ljungbox(res, lags=20)  # 返回统计量和p值，lags为检验的延迟数