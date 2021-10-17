# Python
import pystan
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from matplotlib import pyplot as plt
import json
from prophet.serialize import model_to_json, model_from_json
from sklearn.model_selection import train_test_split
'''
import logging

logging.basicConfig(level='INFO')

mlogger = logging.getLogger('matplotlib')
mlogger.setLevel(logging.WARNING)
'''

with open('model2.json', 'r') as fin:
    m = model_from_json(json.load(fin))  # Load model

# prediction
future = m.make_future_dataframe(periods=8000, freq='5min')
future.tail()
forecast = m.predict(future)


# data
df = pd.read_csv('./3Month.csv')
df = df[["time", "open"]]
# Rename the features: These names are required for the model fitting
df = df.rename(columns={"time": "ds", "open": "y"})
df.head()
#df = pd.DataFrame(df).to_numpy()
df1, df2 = train_test_split(df, test_size=0.28, shuffle=False)
# print(df['y'])
# fig2 = m.plot_components(forecast)
# forecast[['ds', 'yhat']].plot()
# plt.show()
# plt.plot(forecast[['y']], forecast[['yhat']])
plt.set_loglevel('WARNING')
#plt.plot(forecast['ds'], forecast['yhat'])
#plt.plot(df[0], df[1])
# df.plot()
plt.show()
print(type(forecast))
print(type(df))
# print(forecast['ds'])
# print(forecast['yhat'])
