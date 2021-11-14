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

n_estimators=15
random_state=100
max_depth=3
min_samples_leaf=15
subsample=0.33

tree=DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

tree_=DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

mu=np.mean(dataLags, axis=0)
std=np.std(dataLags, axis=0)
normalisedData=(dataLags-mu)/std
testInd=8000

tree.fit(normalisedData[:testInd, :], data['direction'][lags:testInd+lags])
data.loc[:,'predict_Normalised']=pd.Series(tree.predict(normalisedData)).shift(lags+window)

tree_.fit(dataLags[:testInd, :], data['direction'][lags:testInd+lags])
data.loc[:,'predict_No_Normal']=pd.Series(tree_.predict(dataLags)).shift(lags+window)


data.dropna(inplace=True)

print('Without Normalisation:')
print('Training Accuracy')
print(accuracy_score(data['direction'][:testInd], data['predict_No_Normal'][:testInd]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][testInd:], data['predict_No_Normal'][testInd:]))

print('With Normalisation:')
print('Training Accuracy')
print(accuracy_score(data['direction'][:testInd], data['predict_Normalised'][:testInd]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][testInd:], data['predict_Normalised'][testInd:]))

trades_No_Norm=data['predict_No_Normal'].diff()!=0
trades_Normalise=data['predict_Normalised'].diff()!=0
data['strategy_No_Norm']=data['predict_No_Normal']*data['returns']
data['strategy_Normalised']=data['predict_Normalised']*data['returns']
print('No. of trades :')
print(sum(trades_No_Norm))
print(sum(trades_Normalise))
test=data[testInd:]
print('Gross Performance :')
print(test[['returns', 'strategy_No_Norm', 'strategy_Normalised']].sum().apply(np.exp))
test[['returns', 'strategy_No_Norm', 'strategy_Normalised']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

