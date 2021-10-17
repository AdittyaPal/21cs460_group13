import pandas as pd
import numpy as np
from pylab import mpl, plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
raw=pd.read_csv('./gold-Current.csv', index_col=0, parse_dates=True).dropna()
print(raw.info())
'''
data=pd.DataFrame(raw[''])
lags=2
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['font.family']='serif'
data['returns']=np.log(data['close']/data['close'].shift(1))
data.dropna(inplace=True)

x=np.lib.stride_tricks.sliding_window_view(data['returns'], lags)[:-1,:]
y=np.sign(data['returns'][lags:])

model=LogisticRegression(C=0.05, solver='liblinear', multi_class='ovr', random_state=0)
model=model.fit(x, y)
predict=model.predict(x)
print(metrics.confusion_matrix(y, predict))
print(metrics.classification_report(y, predict))


data.dropna(inplace=True)
print(data['predict'].value_counts())
hits=np.sign(data['returns'].iloc[lags:]*data['predict'].iloc[lags:]).value_counts()
print(hits)
print(accuracy_score(data['predict'], np.sign(data['returns'])))
data['strategy']=data['predict']*data['returns']
print(data[['returns', 'strategy']].sum().apply(np.exp))
data[['returns', 'strategy']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
'''
