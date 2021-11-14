import pandas as pd
import numpy as np
from pylab import mpl, plt

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
from sklearn.ensemble import RandomForestClassifier

n_estimators=100
random_state=100
max_depth=3
min_samples_leaf=10
max_samples=0.80

tree=DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
model=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples, min_samples_leaf=min_samples_leaf, random_state=random_state)

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
data.loc[:,'predict_Tree']=pd.Series(tree.predict(normalisedData)).shift(lags+window)
data.loc[:,'predict_RanFor']=pd.Series(model.predict(normalisedData)).shift(lags+window)
data.dropna(inplace=True)

print('Using a single Decision tree:')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict_Tree'][:5000]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict_Tree'][5000:]))

print('Using a Random Forest:')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict_RanFor'][:5000]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict_RanFor'][5000:]))

tradesTree=data['predict_Tree'].diff()!=0
tradesFor=data['predict_RanFor'].diff()!=0
data['strategy_Tree']=data['predict_Tree']*data['returns']
data['strategy_Random_Forest']=data['predict_RanFor']*data['returns']
print('No. of trades :')
print(sum(tradesTree))
print(sum(tradesFor))
test=data[5000:]
print('Gross Performance :')
print(test[['returns', 'strategy_Tree', 'strategy_Random_Forest']].sum().apply(np.exp))
test[['returns', 'strategy_Tree', 'strategy_Random_Forest']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

