# Python
import pystan
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from matplotlib import pyplot as plt
import json
from prophet.serialize import model_to_json, model_from_json
from sklearn.model_selection import train_test_split


with open('model4.json', 'r') as fin:
    m = model_from_json(json.load(fin))  # Load model

# prediction
future = m.make_future_dataframe(periods=21600, freq='1min')
#future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)

'''
# data
df = pd.read_csv('./3Month.csv')
df = df[["time", "open"]]
df = df.rename(columns={"time": "ds", "open": "y"})
df1, df2 = train_test_split(df, test_size=0.28, shuffle=False)
plt.plot(forecast['ds'], forecast['yhat'])
#plt.plot(df["ds"], df["y"])
'''
# forecast.plot()
plt.plot(forecast['ds'], forecast['yhat'])
m.plot(forecast)
plt.show()
print(type(forecast))
# forecast.to_csv('forecast3.csv')
# forecast.csv 15 days prediction
# forecast2.csv 30 days prediction
# forecast3.csv 1 day prediction
# print(type(df))
# print(forecast['ds'])
# print(forecast['yhat'])
