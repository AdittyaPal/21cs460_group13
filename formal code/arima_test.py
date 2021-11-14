'''
Python Class for Vectorised-Backtesting of an ARIMA Based Strategy
'''
import pandas as pd
import numpy as np
from pylab import mpl, plt
import pmdarima as pm

class ARIMABacktester(object):
	'''
	Class for the vectorised backtesting of an ARIMA based strategy
	Instance Variables:
	column : the column of the dataset to be used
	start  : start date for training data
	end    : end date for training data
	amount : amount to be invested
	tc     : proportional transaction costs
	
	Member Functions:
	get_data()     : retrieve the dataset
	fit_model()    : fit a regression model to the data
	run_strategy() : test the model returned by fit_model()
	plot_results() : plot the returns from the adopted strategy
	'''
	def __init__(self, column, start, end, amount, tc):
		self.column=column
		self.start=start
		self.end=end
		self.amount=amount
		self.tc=tc
		self.results=None
		self.get_data()

	def get_data(self):
		''' 
		Retrieves and prepares the DatFrame
		'''
		raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna()
		raw=pd.DataFrame(raw[self.column][self.start:self.end])
		raw['logs']=np.log(raw[self.column])
		raw['returns']=np.log(raw[self.column]/raw[self.column].shift(1))
		self.data=raw.dropna()

	def fit_model(self, start, end):
		''' 
		Implement and fit the perceptron model
		'''
		#ARIMA model fitting
		model=pm.auto_arima(self.data['returns'][start:end], start_p=0, strat_q=0, start_P=0, start_Q=0, d=0, max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False)
		return model

	def run_strategy(self, start_train, end_train, start_test, end_test):
		''' 
		Backtests the trading strategy
		'''		
		self.results=pd.DataFrame(self.data[start_test:end_test])
		model=self.fit_model(start_train, end_train)
		print("Determining optimal paramaters done.")
		predict=np.zeros(end_test-start_test)
		while(end_train<end_test):
			model.fit(self.data['returns'][end_train-200:end_train])
			predict[end_train-start_test]=model.predict(n_periods=1, return_conf_int=False)
			#print('Actual: %.8f, Predicted: %.8f'%(self.data['logs'][end_train+1], predict[end_train-start_test]))
			end_train+=1
		
		print("Prediction done.")
		self.results['predict']=predict
		self.results['predictReturns']=self.results['predict']-self.results['predict'].shift(1)
		self.results['strategy']=np.sign(self.results['predictReturns'])*self.results['returns']
		self.results['creturns']=self.amount*self.results['returns'].cumsum().apply(np.exp)
		self.results['cstrategy']=self.amount*self.results['strategy'].cumsum().apply(np.exp)
		# gross performance of the strategy
		aperf=self.results['cstrategy'].iloc[-1]
        # out-/underperformance of strategy
		operf=aperf-self.results['creturns'].iloc[-1]
        # hit rate
		hits=np.sign(self.results['returns']*self.results['predictReturns']).value_counts()
		print('Gross Performance: %f'%(operf))
		print(hits)

	def plot_results(self):
		''' 
        Plots the cumulative performance of the trading strategy with respect to base returns
		'''
		if self.results is None:
			print('No results to plot yet. Run a strategy.')
		plt.style.use('seaborn')
		mpl.rcParams['savefig.dpi']=300
		mpl.rcParams['font.family']='serif'
		title = 'USD/EUR at 1 day intervals | TC = %.4f, by ARIMA'%(self.tc)
		self.results[['returns', 'predictReturns']].plot(title=title, figsize=(10, 6))
		self.results[['creturns', 'cstrategy']].plot(title=title, figsize=(10, 6))

if __name__ == '__main__':
	lrbt=ARIMABacktester('close', 0, 8000, 1, 0.0)
    # first slice is training and second for prediction anf the last parameter is lags
	lrbt.run_strategy(0, 5000, 5000, 5050)
	lrbt.plot_results()
	plt.show()
