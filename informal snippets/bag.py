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
from sklearn.ensemble import BaggingClassifier

max_depth=3
min_samples_leaf=15
subsample=0.33
n_estimators=100
max_samples=0.9
max_features=0.75
random_state=100

tree=DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
model=BaggingClassifier(base_estimator=tree, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, random_state=random_state)

mu=np.mean(dataLags, axis=0)
std=np.std(dataLags, axis=0)
normalisedData=(dataLags-mu)/std
train=normalisedData[:5000, :]
test=normalisedData[5000: , :]

tree.fit(train, data['direction'][lags:5000+lags])
model.fit(train, data['direction'][lags:5000+lags])
'''
predict=model.predict(train)
print('Training Accuracy')
print(accuracy_score(data['direction'][lags:5000+lags], predict))

test_predict=model.predict(test)
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000+lags:], test_predict))
'''
data.loc[:,'predict_No_Bag']=pd.Series(tree.predict(normalisedData)).shift(lags+window)
data.loc[:,'predict_Bagging']=pd.Series(model.predict(normalisedData)).shift(lags+window)
data.dropna(inplace=True)

print('Without Bagging :')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict_No_Bag'][:5000]))

print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict_No_Bag'][5000:]))

print('With Bagging :')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict_Bagging'][:5000]))

print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict_Bagging'][5000:]))

tradesNoBag=data['predict_No_Bag'].diff()!=0
tradesBag=data['predict_Bagging'].diff()!=0
data['strategy_No_Bag']=data['predict_No_Bag']*data['returns']
data['strategy_Bagging']=data['predict_Bagging']*data['returns']
print('No. of trades :')
print(sum(tradesNoBag))
print(sum(tradesBag))
test=data[5000:]
print('Gross Performance :')
print(test[['returns', 'strategy_No_Bag', 'strategy_Bagging']].sum().apply(np.exp))
test[['returns', 'strategy_No_Bag', 'strategy_Bagging']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
