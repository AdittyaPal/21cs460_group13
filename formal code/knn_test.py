'''
Python Class for Vectorised-Backtesting of
k-Nearest Neighbour Based Strategy
'''
import pandas as pd
import numpy as np
from pylab import mpl, plt
from sklearn.neighbors import KNeighborsClassifier

class LRVectorBacktester(object):
	'''
	Class for the vectorised backtesting of a linear-regression based strategy
	Instance Variables:
	column : the column of the dataset to be used
	start  : start date for training data
	end    : end date for training data
	amount : amount to be invested
	tc     : proportional transaction costs
	
	Member Functions:
	get_data()     : retrieve the dataset
	get_features() : get the data with lags as feature vectors
	fit_model()    : fit a regression model to the data
	run_strategy() : test the model returned by fit_model()
	plot_results() : plot the returns from the adopted strategy
	'''
	def __init__(self, column, start, end, amount, tc):
		self.column = column
		self.start = start
		self.end = end
		self.amount = amount
		self.tc = tc
		self.results = None
		self.get_data()

	def get_data(self):
		''' 
		Retrieves and prepares the DataFrame
		'''
		raw = pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
		raw = pd.DataFrame(raw[self.column][self.start:self.end])
		raw['returns'] = np.log(raw / raw.shift(1))
		window=20
		raw['vol']=raw['returns'].rolling(window).std()
		raw['mom']=raw['returns'].rolling(window).mean()
		raw['sma']=raw['close'].rolling(window).mean()
		raw['min']=raw['close'].rolling(window).min()
		raw['max']=raw['close'].rolling(window).max()
		raw['direction']=np.where(raw['returns']>0, 1, -1)
		self.data = raw.dropna()

	def get_features(self, start, end, lags):
		'''
		Prepare the feature vectors by lag of observations
		'''
		features=['returns', 'vol', 'mom', 'sma', 'min', 'max']
		dataLag=np.lib.stride_tricks.sliding_window_view(self.data[features][start:end], window_shape=lags, axis=0)[:-1,:].reshape((-1, len(features)*lags))
		return dataLag

	def fit_model(self, start, end, lags):
		''' 
		Implement and fit the k-nearest neighbour model
		'''
		dataLagReturn=self.get_features(start, end, lags)
		
		#k nearest neighbours
		model=KNeighborsClassifier(n_neighbors=2)
		model.fit(dataLagReturn, self.data['direction'][lags:end])
		return model

	def run_strategy(self, start_train, end_train, start_test, end_test, lags):
		''' 
		Backtests the trading strategy
		'''
		model=self.fit_model(start_train, end_train, lags)
		self.results=pd.DataFrame(self.data[start_test+lags:end_test])
		testData=self.get_features(start_test, end_test, lags)
		self.results['prediction']=model.predict(testData)
		self.results['strategy']=self.results['prediction']*self.results['returns']
		trades=self.results['prediction'].diff().fillna(0)!=0
		self.results['strategy_tc']=self.results['strategy']-self.tc
		self.results['creturns']=self.amount*self.results['returns'].cumsum().apply(np.exp)
		self.results['cstrategy']=self.amount*self.results['strategy'].cumsum().apply(np.exp)
		# gross performance of the strategy
		aperf=self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
		operf=aperf-self.results['creturns'].iloc[-1]
        # hit rate
		hits=np.sign(self.results['returns']*self.results['prediction']).value_counts()
		print('Gross Performance: %f'%(operf))
		print(hits)
		print('Number of trades : %f'%(sum(trades)))

	def plot_results(self):
		''' 
        Plots the cumulative performance of the trading strategy with respect to base returns
		'''
		if self.results is None:
			print('No results to plot yet. Run a strategy.')
		plt.style.use('seaborn')
		mpl.rcParams['savefig.dpi']=300
		mpl.rcParams['font.family']='serif'
		title = 'USD/EUR at 1 min intervals | TC = %.4f' % (self.tc)
		self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10, 6))

if __name__ == '__main__':
	lrbt=LRVectorBacktester('close', 0, 8000, 1, 0.0)
    # first slice is training and second for prediction anf the last parameter is lags
	lrbt.run_strategy(0, 5000, 5000, 8000, lags=6)
	lrbt.plot_results()
	plt.show()
