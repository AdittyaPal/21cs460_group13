import pandas as pd
import numpy as np
from pylab import mpl, plt

#read the data
raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
data=pd.DataFrame(raw[['time','close']][0:12000])
data['returns']=np.log(data['close']/data['close'].shift(1))

#engineer required features
window=20
data['vol']=data['returns'].rolling(window).std()
data['mom']=data['returns'].rolling(window).mean()
data['sma']=data['close'].rolling(window).mean()
data['min']=data['close'].rolling(window).min()
data['max']=data['close'].rolling(window).max()
data.dropna(inplace=True)

#plotting formatting
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi']=300
mpl.rcParams['font.family']='serif'

#prerare the feature vectors
lags=6
features=['returns', 'vol', 'mom', 'sma', 'min', 'max']
dataLag=np.lib.stride_tricks.sliding_window_view(data['returns'], lags)[:-1,:]
dataLags=np.lib.stride_tricks.sliding_window_view(data[features], window_shape=lags, axis=0)[:-1,:].reshape((-1, len(features)*lags))
data['direction']=np.where(data['returns']>0, 1, -1)
data.dropna(inplace=True)

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#parameters of the decision tree
n_estimators=15
random_state=100
max_depth=3
min_samples_leaf=15
subsample=0.33

#create three separate decision tree models
tree=DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

tree__=DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

tree_=DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

#normalise the data with engineered features
mu=np.mean(dataLags, axis=0)
std=np.std(dataLags, axis=0)
normalisedData=(dataLags-mu)/std
testInd=8000

#fit a decision tree to the data with engineered features
tree.fit(normalisedData[:testInd, :], data['direction'][lags:testInd+lags])
data.loc[:,'predict_Normalised']=pd.Series(tree.predict(normalisedData)).shift(lags+window)

#normalise the data without engineering any extra features
mu=np.mean(dataLag, axis=0)
std=np.std(dataLag, axis=0)
normalised=(dataLag-mu)/std

#fit a decision tree o the data woithout engineered features
tree__.fit(normalised[:testInd, :], data['direction'][lags:testInd+lags])
data.loc[:,'predict_No_Engg']=pd.Series(tree__.predict(normalised)).shift(lags+window)

#fit the un-normalised data, without engineered features to the third decision tree
tree_.fit(dataLags[:testInd, :], data['direction'][lags:testInd+lags])
data.loc[:,'predict_No_Normal']=pd.Series(tree_.predict(dataLags)).shift(lags+window)

data.dropna(inplace=True)

#print the results of the tree decision trees
print('Without Normalisation:')
print('Training Accuracy')
print(accuracy_score(data['direction'][:testInd], data['predict_No_Normal'][:testInd]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][testInd:], data['predict_No_Normal'][testInd:]))

print('Without Feature Engineering:')
print('Training Accuracy')
print(accuracy_score(data['direction'][:testInd], data['predict_No_Engg'][:testInd]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][testInd:], data['predict_No_Engg'][testInd:]))

print('With Normalisation:')
print('Training Accuracy')
print(accuracy_score(data['direction'][:testInd], data['predict_Normalised'][:testInd]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][testInd:], data['predict_Normalised'][testInd:]))

#calculate and print the number of trades 
trades_No_Norm=data['predict_No_Normal'].diff()!=0
trades_No_Engg=data['predict_No_Engg'].diff()!=0
trades_Normalise=data['predict_Normalised'].diff()!=0
print('No. of trades :')
print(sum(trades_No_Norm))
print(sum(trades_No_Engg))
print(sum(trades_Normalise))

#determine the performance of each model
data['strategy_Vanilla']=data['predict_No_Normal']*data['returns']
data['strategy_Normalised_Only']=data['predict_No_Engg']*data['returns']
data['strategy_Normalised+FeatureEngineer']=data['predict_Normalised']*data['returns']

test=data[testInd:]
print('Gross Performance :')
print(test[['returns', 'strategy_Vanilla', 'strategy_Normalised_Only', 'strategy_Normalised+FeatureEngineer']].sum().apply(np.exp))
#plot results
test[['returns', 'strategy_Vanilla', 'strategy_Normalised_Only', 'strategy_Normalised+FeatureEngineer']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
