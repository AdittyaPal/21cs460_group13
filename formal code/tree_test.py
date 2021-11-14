import pandas as pd
import numpy as np
from pylab import mpl, plt

#read the data
raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
data=pd.DataFrame(raw[['time','close']][0:12000])

#engineer required features
data['returns']=np.log(data['close']/data['close'].shift(1))
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
dataLags=np.lib.stride_tricks.sliding_window_view(data[features], window_shape=lags, axis=0)[:-1,:].reshape((-1, len(features)*lags))
data['direction']=np.where(data['returns']>0, 1, -1)
data.dropna(inplace=True)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#parameters of the decision tree
min_samples_leaf=15
random_state=100

#construct decision trees of varying heights
tree2=DecisionTreeClassifier(random_state=random_state, max_depth=2, min_samples_leaf=min_samples_leaf)
tree3=DecisionTreeClassifier(random_state=random_state, max_depth=3, min_samples_leaf=min_samples_leaf)
tree4=DecisionTreeClassifier(random_state=random_state, max_depth=4, min_samples_leaf=min_samples_leaf)
tree5=DecisionTreeClassifier(random_state=random_state, max_depth=5, min_samples_leaf=min_samples_leaf)
tree10=DecisionTreeClassifier(random_state=random_state, max_depth=10, min_samples_leaf=min_samples_leaf)

#normalise the data
mu=np.mean(dataLags, axis=0)
std=np.std(dataLags, axis=0)
normalisedData=(dataLags-mu)/std
train=normalisedData[:5000, :]
test=normalisedData[5000: , :]

#fit each tree of different height with the train data
tree2.fit(train, data['direction'][lags:5000+lags])
tree3.fit(train, data['direction'][lags:5000+lags])
tree4.fit(train, data['direction'][lags:5000+lags])
tree5.fit(train, data['direction'][lags:5000+lags])
tree10.fit(train, data['direction'][lags:5000+lags])

#prdict the labels for the testing data
data.loc[:,'predict2']=pd.Series(tree2.predict(normalisedData)).shift(lags+window)
data.loc[:,'predict3']=pd.Series(tree3.predict(normalisedData)).shift(lags+window)
data.loc[:,'predict4']=pd.Series(tree4.predict(normalisedData)).shift(lags+window)
data.loc[:,'predict5']=pd.Series(tree5.predict(normalisedData)).shift(lags+window)
data.loc[:,'predict10']=pd.Series(tree10.predict(normalisedData)).shift(lags+window)

data.dropna(inplace=True)

#print the accuracy
print('Depth of tree : 2 ')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict2'][:5000]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict2'][5000:]))

print('Depth of tree : 3 ')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict3'][:5000]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict3'][5000:]))

print('Depth of tree : 4 ')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict4'][:5000]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict4'][5000:]))

print('Depth of tree : 5 ')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict5'][:5000]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict5'][5000:]))

print('Depth of tree : 10 ')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict10'][:5000]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict10'][5000:]))

#calculate the number of trades
NoOfTrades=(data[['predict2', 'predict3', 'predict4', 'predict5', 'predict10']].diff()!=0).sum()
print('No. of trades :')
print(NoOfTrades)

#run the performance of each model
data['Tree Depth : 2']=data['predict2']*data['returns']
data['Tree Depth : 3']=data['predict3']*data['returns']
data['Tree Depth : 4']=data['predict4']*data['returns']
data['Tree Depth : 5']=data['predict5']*data['returns']
data['Tree Depth : 10']=data['predict10']*data['returns']

test=data[5000:]
print('Gross Performance :')
print(test[['returns', 'Tree Depth : 2', 'Tree Depth : 3', 'Tree Depth : 4', 'Tree Depth : 5', 'Tree Depth : 10']].sum().apply(np.exp))
#plot the performance
test[['returns', 'Tree Depth : 2', 'Tree Depth : 3', 'Tree Depth : 4', 'Tree Depth : 5', 'Tree Depth : 10']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
