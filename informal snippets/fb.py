# Python
import pystan
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from matplotlib import pyplot as plt
import json
from prophet.serialize import model_to_json, model_from_json


df = pd.read_csv('.\OneMonthData.csv')
df.head()

# Select Date and Price
df = df[["time", "open"]]
# Rename the features: These names are required for the model fitting
df = df.rename(columns={"time": "ds", "open": "y"})
df.head()

# fit model
m = Prophet()
fit = m.fit(df)

# prediction
future = m.make_future_dataframe(periods=15)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = plot_plotly(fit, forecast)
fig2 = m.plot_components(forecast)
fig3 = m.plot(forecast)
plt.show()

'''
fbp.fit(df)
fut = fbp.make_future_dataframe(periods=365)
forecast = fbp.predict(fut)

from fbprophet.plot import plot_plotly, plot_components_plotly
# A better plot than the simple matplotlib
plot_plotly(fbp, forecast)

# print(df.head())
'''
