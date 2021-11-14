from backtest import BacktestBase
import numpy as np
import pandas as pd
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

class BacktestLongOnly(BacktestBase):

	def get_features(self, start, end, lags):
		'''
		Prepare the feature vectors by lag of observations
		'''
		dataLag=np.lib.stride_tricks.sliding_window_view(self.data['returns'][start:end], lags)[:-1,:]
		return dataLag

	def fit_model(self, start, end, lags):
		''' 
		Implement and fit the regression model
		'''
		dataLagReturn=self.get_features(start, end, lags)
		# linear regression
		regParams=np.linalg.lstsq(dataLagReturn, self.data['returns'][lags:end], rcond=None)[0]
		return regParams

	def run_regression_strategy(self, start_train, end_train, start_test, end_test, lags):
		text=f'\n\nRunning linear Regression strategy | lags={lags}'
		text+=f'\nfixed costs {self.ftc} | '
		text+=f'proportional costs {self.ptc}'
		print(text)          #print the parameters
		print('='*55)
		self.position=0      #initially the position is neutral
		self.trades=0        #no trades have been exeecuted yet
		self.amount=self.initialAmount  #reset initial capital
		reg=self.fit_model(start_train, end_train, lags)
		for bar in range(start_test, end_test):   #loop over each available tick value
			if self.position==0:                  #check if trades have been placed
				if (np.dot(self.data['returns'][bar-lags:bar], reg)>0):  #if the price is signalled to increase, then a buy order is placed
					self.place_buy_order(bar, amount=self.amount)   #buy with the initial cash balance
					self.position=1  #set the position to long
			elif self.position==1:   #if assets are held
				if (np.dot(self.data['returns'][bar-lags:bar], reg)<0):  #and there is a signal that prices would decrease
					self.place_sell_order(bar, units=self.units) #sell the held currency
					self.position=0  #set the market to neutral
		self.close_out(bar)  #cloase out any remaining open positions

if __name__ == '__main__':
	test_long=BacktestLongOnly('close', 0, 6000, 1000, verbose=False)
	test_long.run_regression_strategy(0, 5000, 5000, 6000, 7)
