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

data['strategy']=data['predict']*data['returns']

#if an intial balance of 20000 (default of demo account) then equity=initialBal/leverage
equity=100
data['equity']=data['strategy'].cumsum().apply(np.exp)*equity
data['cummax']=data['equity'].cummax()
data['drawdown']=data['cummax']-data['equity']
print('Maximum Drawdown: %0.5f'%(data['drawdown'].max()))
temp=data['drawdown'][data['drawdown']==0]
periods=(temp.index[1:]-temp.index[:-1])
longest_drawdown=periods.max()
print(longest_drawdown)
test=data[6000:]

leverage=10
percentiles=[0.01, 0.1, 1.0, 2.5, 5.0, 10.0]
data['levReturn']=np.log(data['equity']/data['equity'].shift(1))*leverage
data.dropna(inplace=True)
val_at_risk=np.percentile(data['levReturn']*equity*leverage, percentiles)
print('{}              {}'.format('Confidence Level', 'Value-at-Risk'))
print(50*'-')
for pair in zip(percentiles, val_at_risk):
	print('{:15.2f} {:20.3f}'.format(100-pair[0], -pair[1]))

data['levReturn'].hist(bins=50, figsize=(10, 8))
ax=data['levReturn'].plot(kind='kde', figsize=(10, 8), xlim=(-0.005, 0.005))

quant_1, quant_5, quant_10=data['levReturn'].quantile(0.01), data['levReturn'].quantile(0.05), data['levReturn'].quantile(0.1)
quants=[[quant_1, 0.6, 0.10], [quant_5, 0.8, 0.30], [quant_10, 1, 0.50]]
for i in quants:
    plt.axvline(i[0], c='r', alpha = i[1], ymax = i[2], linestyle = ":")
plt.show()

