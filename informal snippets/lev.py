import pandas as pd
import numpy as np
from pylab import mpl, plt

raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
data=pd.DataFrame(raw[['close']][0:7500])
data['returns']=np.log(data['close']/data['close'].shift(1))
window=20
data['vol']=data['returns'].rolling(window).std()
data['mom']=data['returns'].rolling(window).mean()
data['sma']=data['close'].rolling(window).mean()
data['min']=data['close'].rolling(window).min()
data['max']=data['close'].rolling(window).max()
data.dropna(inplace=True)

plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['font.family']='serif'

lags=6
features=['returns', 'vol', 'mom', 'sma', 'min', 'max']
dataLags=np.lib.stride_tricks.sliding_window_view(data[features], window_shape=lags, axis=0)[:-1,:].reshape((-1, len(features)*lags))
data['direction']=np.where(data['returns']>0, 1, -1)
data.dropna(inplace=True)

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

n_estimators=15
random_state=100
max_depth=3
min_samples_leaf=15
subsample=0.33

tree=DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
model=AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators, random_state=random_state)

mu=np.mean(dataLags, axis=0)
std=np.std(dataLags, axis=0)
normalisedData=(dataLags-mu)/std
train=normalisedData[:6000, :]

model.fit(train, data['direction'][lags:6000+lags])
data.loc[:,'predict']=pd.Series(model.predict(normalisedData)).shift(lags+window)
data.dropna(inplace=True)

equs= []

def leverage(f, equs):
	equ='leverage_{:.2f}'.format(f)
	equs.append(equ)
	cap = 'capital_{:.2f}'.format(f)
	data[equ]=data['returns']*f

leverage(1, equs)	
leverage(2, equs)
leverage(5, equs)
leverage(10, equs)
leverage(50, equs)

data[equs].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
	
