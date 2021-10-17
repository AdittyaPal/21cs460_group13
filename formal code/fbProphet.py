# Python
import pystan
import pandas as pd
from prophet import Prophet
from matplotlib import pyplot as plt
import json
from prophet.serialize import model_to_json, model_from_json

df = pd.read_csv('./OneMonthData.csv')
df = df[["time", "open"]]
# Rename the features: These names are required for the model fitting
df = df.rename(columns={"time": "ds", "open": "y"})
df.head()

# fit model
# scale 0.01
m = Prophet(changepoint_prior_scale=0.01)
fit = m.fit(df)

# As fitting and prediction both take time so it is better to save
# fit model and use that for different predictions
with open('model.json', 'w') as fout:
    json.dump(model_to_json(fit), fout)  # Save model

with open('model.json', 'r') as fin:
    m = model_from_json(json.load(fin))  # Load model

# prediction
future = m.make_future_dataframe(periods=21600, freq='1min')
future.tail()
forecast = m.predict(future)

# forecast.plot()
plt.plot(forecast['ds'], forecast['yhat'])
m.plot(forecast)
fig2 = m.plot_components(forecast)
plt.show()
