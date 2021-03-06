import pandas as pd
import numpy as np
from pylab import mpl, plt
import pmdarima as pm

raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
data=pd.DataFrame(raw[['time','close']][0:12000])
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
testInd=5000
train=normalisedData[:testInd, :]
test=normalisedData[testInd: , :]

model.fit(train, data['direction'][lags:testInd+lags])
tree.fit(train, data['direction'][lags:testInd+lags])
'''
predict=model.predict(train)
print('Training Accuracy')
print(accuracy_score(data['direction'][lags:5000+lags], predict))

test_predict=model.predict(test)
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000+lags:], test_predict))
'''
data.loc[:,'predict_with_boost']=pd.Series(model.predict(normalisedData)).shift(lags+window)
data.loc[:,'predict_no_boost']=pd.Series(tree.predict(normalisedData)).shift(lags+window)
data.dropna(inplace=True)

print('Without Boosting:')
print('Training Accuracy')
print(accuracy_score(data['direction'][:testInd], data['predict_no_boost'][:testInd]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][testInd:], data['predict_no_boost'][testInd:]))

print('With Boosting:')
print('Training Accuracy')
print(accuracy_score(data['direction'][:testInd], data['predict_with_boost'][:testInd]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][testInd:], data['predict_with_boost'][testInd:]))

trades_with_boost=data['predict_with_boost'].diff()!=0
trades_no_boost=data['predict_no_boost'].diff()!=0
data['strategy_no_boost']=data['predict_no_boost']*data['returns']
data['strategy_boost']=data['predict_with_boost']*data['returns']
print('No. of trades :')
print(sum(trades_with_boost))
print(sum(trades_no_boost))
test=data[testInd:]
print('Gross Performance :')
print(test[['returns', 'strategy_no_boost', 'strategy_boost']].sum().apply(np.exp))
test[['returns', 'strategy_no_boost', 'strategy_boost']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

