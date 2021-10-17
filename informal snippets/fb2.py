# Python
import pystan
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from matplotlib import pyplot as plt
import json
from prophet.serialize import model_to_json, model_from_json
from sklearn.model_selection import train_test_split

df = pd.read_csv('./OneMonthData.csv')
df = df[["time", "open"]]
# Rename the features: These names are required for the model fitting
df = df.rename(columns={"time": "ds", "open": "y"})
df.head()

df1, df2 = train_test_split(df, test_size=0.05, shuffle=False)

# fit model
# scale 0.01
# m = Prophet(changepoint_prior_scale=0.01)
m = Prophet()
fit = m.fit(df1)

with open('model4.json', 'w') as fout:
    json.dump(model_to_json(fit), fout)  # Save model

# print(df1)
# print(df2)
# model for 1 month data
# model2 for 3 month data, 5mint
# model3 for 4 year
# model4 1month for 29 days
