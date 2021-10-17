import pandas as pd
import numpy as np
from pylab import mpl, plt
from math import nan
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

raw=pd.read_csv('./gold1year.csv', index_col=0, parse_dates=True).dropna()

data=pd.DataFrame(raw['close'])
data.rename(columns={'close': 'price'}, inplace=True)
data['return']=np.log(data['price']/data['price'].shift(1))
data.dropna(inplace=True)
print(data.info())

lags=5

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['font.family']='serif'

x=np.lib.stride_tricks.sliding_window_view(data['return'], lags)[:-1,:]
y=np.sign(data['return'][lags:])

data=data[lags:]
model=LogisticRegression(C=1e7, max_iter=1000)
model=model.fit(x, y)
data['predict']=model.predict(x)
data.dropna(inplace=True)
data['strategy']=data['predict']*data['return'][lags:]
print(accuracy_score(data['predict'], np.sign(data['return'])))
print(confusion_matrix(np.sign(data['return']), data['predict']))
print(classification_report(np.sign(data['return']), data['predict']))
print(data[['return', 'strategy']].sum().apply(np.exp))
data[['return', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

