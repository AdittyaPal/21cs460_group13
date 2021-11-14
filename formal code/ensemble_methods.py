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
dataLags=np.lib.stride_tricks.sliding_window_view(data[features], window_shape=lags, axis=0)[:-1,:].reshape((-1, len(features)*lags))
data['direction']=np.where(data['returns']>0, 1, -1)
data.dropna(inplace=True)

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

#parameters of the decision tree
max_depth=3
min_samples_leaf=15
random_state=100

#create a decision tree model with the above listed parameters
tree=DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

#create a model by bagging
model=BaggingClassifier(base_estimator=tree, n_estimators=100, max_samples=0.9, max_features=0.75, random_state=random_state)

#createa second model by boosting
model_=AdaBoostClassifier(base_estimator=tree, n_estimators=15, random_state=random_state)

#create a third model by using a random forest
model__=RandomForestClassifier(n_estimators=100, max_depth=max_depth, max_samples=0.8, min_samples_leaf=10, random_state=random_state)

#normalise the data with engineered features
mu=np.mean(dataLags, axis=0)
std=np.std(dataLags, axis=0)
normalisedData=(dataLags-mu)/std
train=normalisedData[:5000, :]
test=normalisedData[5000: , :]

#fit the normalised training data to each model using a different ensemble method
tree.fit(train, data['direction'][lags:5000+lags])
model.fit(train, data['direction'][lags:5000+lags])
model_.fit(train, data['direction'][lags:5000+lags])
model__.fit(train, data['direction'][lags:5000+lags])

#predict the labels for the testing data for each emsemble method
data.loc[:,'predict_Tree']=pd.Series(tree.predict(normalisedData)).shift(lags+window)
data.loc[:,'predict_Bagging']=pd.Series(model.predict(normalisedData)).shift(lags+window)
data.loc[:,'predict_Boost']=pd.Series(model_.predict(normalisedData)).shift(lags+window)
data.loc[:,'predict_RanFor']=pd.Series(model__.predict(normalisedData)).shift(lags+window)
data.dropna(inplace=True)

#print the accuracy of the emsemble methods
print('Using single Decision tree:')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict_Tree'][:5000]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict_Tree'][5000:]))

print('With Bagging :')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict_Bagging'][:5000]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict_Bagging'][5000:]))

print('With Boosing:')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict_Boost'][:5000]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict_Boost'][5000:]))

print('Using a Random Forest:')
print('Training Accuracy')
print(accuracy_score(data['direction'][:5000], data['predict_RanFor'][:5000]))
print('Testing Accuracy')
print(accuracy_score(data['direction'][5000:], data['predict_RanFor'][5000:]))

#calculate and print the number of trades
tradesTree=data['predict_Tree'].diff()!=0
tradesBag=data['predict_Bagging'].diff()!=0
tradesBoost=data['predict_Boost'].diff()!=0
tradesFor=data['predict_RanFor'].diff()!=0
print('No. of trades :')
print(sum(tradesTree))
print(sum(tradesBag))
print(sum(tradesBoost))
print(sum(tradesFor))

#determine the performance of each model
data['Decision_Tree']=data['predict_Tree']*data['returns']
data['Bagging']=data['predict_Bagging']*data['returns']
data['Boosting']=data['predict_Boost']*data['returns']
data['Random_Forest']=data['predict_RanFor']*data['returns']
test=data[5000:]
print('Gross Performance :')
#plot results
print(test[['returns', 'Decision_Tree', 'Bagging', 'Boosting', 'Random_Forest']].sum().apply(np.exp))
test[['returns', 'Decision_Tree', 'Bagging', 'Boosting', 'Random_Forest']].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()
