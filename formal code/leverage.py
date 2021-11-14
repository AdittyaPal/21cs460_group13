import pandas as pd
import numpy as np
from pylab import mpl, plt

#read the data
raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
data=pd.DataFrame(raw[['close']][0:7500])
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
from sklearn.ensemble import AdaBoostClassifier

#parameters of the decision tree
n_estimators=15
random_state=100
max_depth=3
min_samples_leaf=15
subsample=0.33

#construct a decision trees
tree=DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
#and fit a boosting ensemble method on it
model=AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators, random_state=random_state)

#normalise the data
mu=np.mean(dataLags, axis=0)
std=np.std(dataLags, axis=0)
normalisedData=(dataLags-mu)/std
train=normalisedData[:6000, :]

#fit the normalised training data on the boosted decision tree
model.fit(train, data['direction'][lags:6000+lags])
#and predict labels for the testing data
data.loc[:,'predict']=pd.Series(model.predict(normalisedData)).shift(lags+window)
data.dropna(inplace=True)

#to visuvalise the effect of leverage
equs= []
#define a function that calculates the leveraged returns when passed the leverage f
'''
Parameters
------------------------------
f    : leverage for which return have to be calculated
equs : python list to store the leverages for which returns have been calculated
''' 
def leverage(f, equs):
	equ='leverage_{:.2f}'.format(f)
	equs.append(equ)
	cap = 'capital_{:.2f}'.format(f)
	data[equ]=data['returns']*f

leverage(1, equs)	 #calculates returns without leverage
leverage(2, equs)    #calculates returns with leverage 2, doubles returns
leverage(5, equs)    #calculates returns with leverage 5
leverage(10, equs)   #calculates returns with leverage 10
leverage(50, equs)   #calculates returns with leverage 50

#plot the effect of leverage on returns
data[equs].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.show()

trades=data['predict'].diff()!=0           #calculates the positions where trades take place
data['strategy']=data['predict']*data['returns']

#determine the opimal leverage using the kelly criterion
mean=data[['returns', 'strategy']].mean()*6000
var=data[['returns', 'strategy']].var()*6000
vol=var**0.5
print('Optimal Leverage by full Kelly criterion:')
print(mean/var)
print('Optimal Leverage by half Kelly criterion:')
print(mean/vol)

#leverage = 10 is the opimal determined
#if an intial balance of 1000 (default of demo account) then equity=initialBal/leverage
equity=100
#calculate the equity or the waelth after trading, equity=initialBal * returns
data['equity']=data['strategy'].cumsum().apply(np.exp)*equity
#determine the highs of the equity due the the adpoing of the trading startegy
data['cummax']=data['equity'].cummax()
#dropdown is the difference between the previous high (calculated by cummax()) and the current returns
data['drawdown']=data['cummax']-data['equity']
#determine the largest difference: maximum drawdown
print('Maximum Drawdown: %0.5f'%(data['drawdown'].max()))
#determine the timestamps where new highs are reached by the trading strategy
temp=data['drawdown'][data['drawdown']==0]
#determine the intervening period between two consecutive highs
periods=(temp.index[1:]-temp.index[:-1])
#longest period between two consecutive highs of the returns is the longest drawdown
longest_drawdown=periods.max()
print(longest_drawdown)

test=data[6000:]
test[['equity', 'cummax']].plot(figsize=(10, 6))          #plot the equity and highs reached
plt.axvline(data['drawdown'].idxmax(), c='r', alpha=0.5)  #plot the position wherer maximum drawdown occurs
plt.show()


leverage=10
percentiles=[0.01, 0.1, 1.0, 2.5, 5.0, 10.0]   #percentiles whose values are to be calculated
#calculate the leveraged returns - log of ratio of wealths scaled by the leverage
data['levReturn']=np.log(data['equity']/data['equity'].shift(1))*leverage
data.dropna(inplace=True)
#determine the postion of the requested percentiles
val_at_risk=np.percentile(data['levReturn']*equity*leverage, percentiles)
print('{}              {}'.format('Confidence Level', 'Value-at-Risk'))
print(50*'-')
for pair in zip(percentiles, val_at_risk):
	print('{:15.2f} {:20.3f}'.format(100-pair[0], -pair[1]))  #convert percentile values to confidence values

#plot a histogram of the returns (distribuion of the returns)
data['levReturn'].hist(bins=50, figsize=(10, 8))
#plot the kernel density function- the probablity density function of the returns
ax=data['levReturn'].plot(kind='kde', figsize=(10, 8), xlim=(-0.005, 0.005))

#calculate the positions where the quantiles occur
quant_1, quant_5, quant_10=data['levReturn'].quantile(0.01), data['levReturn'].quantile(0.05), data['levReturn'].quantile(0.1)   
quants=[[quant_1, 0.6, 0.10], [quant_5, 0.8, 0.30], [quant_10, 1, 0.50]]
#plot the quantiles
for i in quants:
    plt.axvline(i[0], c='r', alpha = i[1], ymax = i[2], linestyle = ":")
plt.show()
