'''
Python Class for Vectorised-Backtesting of
Linear-Regression Based Strategy
'''
import pandas as pd
import numpy as np
from pylab import mpl, plt

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
		Retrieves and prepares the DatFrame
		'''
		raw = pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
		raw = pd.DataFrame(raw[self.column][self.start:self.end])
		raw['returns'] = np.log(raw / raw.shift(1))
		self.data = raw.dropna()

	def get_features(self, start, end, lags):
		'''
		Prepare the feature vectors by lag of observations
		'''
		dataLag=np.lib.stride_tricks.sliding_window_view(self.data['returns'][start:end], lags)[:-1,:]
		return dataLag
		
	def get_features2(self, start, end, lags, window):
		'''
		Prepare the feature vectors by lag of observations
		'''
		data_=pd.DataFrame(self.data[start:end])
		data_['vol']=data_['returns'].rolling(window).std()
		data_['mom']=data_['returns'].rolling(window).mean()
		data_['sma']=data_['close'].rolling(window).mean()
		data_['min']=data_['close'].rolling(window).min()
		data_['max']=data_['close'].rolling(window).max()
		data_.dropna(inplace=True)
		features=['returns', 'vol', 'mom', 'sma', 'min', 'max']
		dataLags=np.lib.stride_tricks.sliding_window_view(data_[features], window_shape=lags, axis=0)[:-1,:].reshape((-1, len(features)*lags))
		return dataLags

	def fit_model(self, start, end, lags):
		''' 
		Implement and fit the regression model
		'''
		dataLagReturn=self.get_features(start, end, lags)
		# linear regression
		regParams=np.linalg.lstsq(dataLagReturn, self.data['returns'][lags:end], rcond=None)[0]
		return regParams
		
	def fit_model2(self, start, end, lags, window):
		''' 
		Implement and fit the regression model
		'''
		dataLagReturn=self.get_features2(start, end, lags, window)
		# linear regression
		regParams=np.linalg.lstsq(dataLagReturn, self.data['returns'][window+lags-1:end], rcond=None)[0]
		return regParams

	def run_strategy(self, start_train, end_train, start_test, end_test, lags, window):
		''' 
		Backtests the trading strategy
		'''
		reg=self.fit_model(start_train, end_train, lags)
		self.results=pd.DataFrame(self.data[start_test+lags:end_test])
		testData=self.get_features(start_test, end_test, lags)
		self.results['predictRate']=np.dot(testData, reg)
		self.results['prediction']=np.sign(self.results['predictRate'])
		self.results['strategy']=self.results['prediction']*self.results['returns']
		trades=self.results['prediction'].diff().fillna(0)!=0
		self.results['strategy_tc']=self.results['strategy']-self.tc
		self.results['creturns']=self.amount*self.results['returns'].cumsum().apply(np.exp)
		self.results['cstrategy']=self.amount*self.results['strategy'].cumsum().apply(np.exp)
		self.results['cstrategy_tc']=self.amount*self.results['strategy_tc'].cumsum().apply(np.exp)
		# gross performance of the strategy
		aperf=self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
		operf=aperf-self.results['creturns'].iloc[-1]
        # hit rate
		hits=np.sign(self.results['returns']*self.results['prediction']).value_counts()
		print('Gross Performance: %f'%(operf))
		print(hits)
		
		reg_fv=self.fit_model2(start_train, end_train, lags, window)
		testData_fv=self.get_features2(start_test, end_test, lags, window)
		self.results.loc[:,'predictRate_fv']=pd.Series(np.dot(testData_fv, reg_fv)).shift(lags+window)
		self.results.dropna(inplace=True)
		self.results['prediction_fv']=np.sign(self.results['predictRate_fv'])
		self.results['strategy_fv']=self.results['prediction_fv']*self.results['returns']
		trades_fv=self.results['prediction_fv'].diff().fillna(0)!=0
		self.results['cstrategy_fv']=self.amount*self.results['strategy_fv'].cumsum().apply(np.exp)
		# gross performance of the strategy
		#aperf_fv=self.results['cstrategy_fv'].iloc[-1]
        # out-/underperformance of strategy
		#operf_fv=aperf_fv-self.results['creturns'].iloc[-1]
        # hit rate
		hits_fv=np.sign(self.results['returns']*self.results['prediction_fv']).value_counts()
		#print('Gross Performance: %f'%(operf_fv))
		print(hits_fv)

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
		#self.results[['creturns', 'cstrategy', 'cstrategy_fv']].plot(title=title, figsize=(10, 6))
		#self.results[['returns', 'predictRate']].plot(title=title, figsize=(10, 6))

if __name__ == '__main__':
	lrbt=LRVectorBacktester('close', 0, 8000, 1, 0.0)
    # first slice is training and second for prediction anf the last parameter is lags
	lrbt.run_strategy(0, 5000, 5000, 8000, lags=5, window=20)
	lrbt.plot_results()
	plt.show()