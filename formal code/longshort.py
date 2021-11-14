from backtest import BacktestBase
import numpy as np
import pandas as pd
from pylab import mpl, plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

class BacktestLongShort(BacktestBase):

	def fit_model(self, start, end, lags):
		''' 
		Implement and fit the decision tree model
		'''
		window=20
		self.data['vol']=self.data['returns'].rolling(window).std()
		self.data['mom']=self.data['returns'].rolling(window).mean()
		self.data['sma']=self.data['close'].rolling(window).mean()
		self.data['min']=self.data['close'].rolling(window).min()
		self.data['max']=self.data['close'].rolling(window).max()
		self.data.dropna(inplace=True)
		lags=6
		features=['returns', 'vol', 'mom', 'sma', 'min', 'max']
		dataLags=np.lib.stride_tricks.sliding_window_view(self.data[features], window_shape=lags, axis=0)[:-1,:].reshape((-1, len(features)*lags))
		self.data['direction']=np.where(self.data['returns']>0, 1, -1)
		self.data.dropna(inplace=True)
		n_estimators=100
		max_samples=1.0
		max_features=0.75
		random_state=100
		max_depth=4
		min_samples_leaf=15
		subsample=0.33

		tree=DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
		model1=BaggingClassifier(base_estimator=tree, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, random_state=random_state)
		model2=AdaBoostClassifier(base_estimator=tree, n_estimators=15, random_state=random_state)
		model3=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_samples=max_samples, min_samples_leaf=min_samples_leaf, random_state=random_state)
		model4=KNeighborsClassifier(n_neighbors=2)
		model5=HistGradientBoostingClassifier(max_iter=100, learning_rate=1.0, max_depth=1, random_state=100)
				
		mu=np.mean(dataLags, axis=0)
		std=np.std(dataLags, axis=0)
		normalisedData=(dataLags-mu)/std
		train=normalisedData[:5000, :]
		test=normalisedData[5000: , :]
		
		model1.fit(train, self.data['direction'][lags:5000+lags])
		model2.fit(train, self.data['direction'][lags:5000+lags])
		model3.fit(train, self.data['direction'][lags:5000+lags])
		model4.fit(train, self.data['direction'][lags:5000+lags])
		model5.fit(train, self.data['direction'][lags:5000+lags])
		return model1, model2, model3, model4, model5, train
		#return model, train
		
	def go_long(self, bar, units=None, amount=None):
		'''
		method to buy a certain number of units of currency
		'''
		if self.position==-1:      #if no currency units have been bought
			self.place_buy_order(bar, units=-self.units)    #it is closed by buying the required number of units of currency
		if units:     #if the number of units are given
			self.place_buy_order(bar, units=units)   #the number of units is bought
		elif amount:                       #else if the amount is specified
			if amount=='all':
				amount=self.amount
			self.place_buy_order(bar, amount=amount)   #the possible units of currency with that amount is bought
	
	def go_short(self, bar, units=None, amount=None):
		if self.position==1:
			self.place_sell_order(bar, units=self.units)
		if units:
			self.place_sell_order(bar, units=units)
		elif amount:
			if amount=='all':
				amount=self.amount
			self.place_sell_order(bar, amount=amount)	

	def run_regression_strategy(self, start_train, end_train, start_test, end_test, lags):
		text=f'\n\nRunning linear Regression strategy | lags={lags}'
		text+=f'\nfixed costs {self.ftc} | '
		text+=f'proportional costs {self.ptc}'
		print(text)          #print the parameters
		print('='*55)
		self.units=0
		self.position=0      #initially the position is neutral
		self.trades=0        #no trades have been exeecuted yet
		self.amount=self.initialAmount  #reset initial capital
		self.sl=0.0004
		self.tsl=0.0004
		self.tp=0.0025
		model1, model2, model3, model4, model5, train=self.fit_model(start_train, end_train, lags)
		#model, train=self.fit_model(start_train, end_train, lags)
		
		for bar in range(start_test, end_test):
			if self.trades==0 and self.verbose==True:
				print(50*'_')
				print(f'{bar} | ---START TESTING---')
				self.print_balance(bar)
				print(50*'_')
			date, price=self.get_price(bar)
			if self.sl!=0 and self.position!=0:
				change=(price-self.entry_price)/self.entry_price
				if self.position==1 and change<-self.sl:
					if self.verbose==True:
						print(50*'-')
						print('--- STOP LOSS (LONG | -{self.sl}) ---')
					self.go_short(bar-1, units=self.units)
					self.position=-1
				elif self.position==-1 and change>self.sl:
					if self.verbose==True:
						print(50*'-')
						print('--- STOP LOSS (SHORT | -{self.sl}) ---')
					self.go_long(bar-1, units=self.units)
					self.position=1
			if self.tsl!=0 and self.position!=0:
				self.max_price=max(self.max_price, price)
				self.min_price=min(self.min_price, price)
				if self.position==1 and (price-self.max_price)/self.entry_price<-self.tsl:
					if self.verbose==True:
						print(50*'-')
						print('--- TRAILING STOP LOSS (LONG | -{self.tsl}) ---')
					self.go_short(bar-1, units=self.units)
					self.position=-1
				elif self.position==-1 and (self.min_price-price)/self.entry_price<-self.tsl:
					if self.verbose==True:
						print(50*'-')
						print('--- TRAILING STOP LOSS (SHORT | -{self.tsl}) ---')
					self.go_long(bar-1, units=self.units)
					self.position=1
			if self.tp!=0 and self.position!=0:
				change=(price-self.entry_price)/self.entry_price
				if self.position==1 and change>self.tp:
					if self.verbose==True:
						print(50*'-')
						print('--- TAKE PROFIT (LONG | -{self.sl}) ---')
					self.go_short(bar-1, units=self.units)
					self.position=-1
				elif self.position==-1 and change<-self.tp:
					if self.verbose==True:
						print(50*'-')
						print('--- TAKE PROFIT (SHORT | -{self.sl}) ---')
					self.go_long(bar-1, units=self.units)
					self.position=1
			action1=model1.predict(train[bar-start_test, :].reshape(1, -1))
			action2=model2.predict(train[bar-start_test, :].reshape(1, -1))
			action3=model3.predict(train[bar-start_test, :].reshape(1, -1))
			action4=model4.predict(train[bar-start_test, :].reshape(1, -1))
			action5=model5.predict(train[bar-start_test, :].reshape(1, -1))
			action=0
			if action1==1 or action2==1 or action3==1 or action4==1 or action5==1:
				action=1
			#action=model.predict(train[bar-start_test, :].reshape(1, -1))
			if self.position==-1 or self.position==0:
				if action==1:
					if self.verbose==True:
						print(50*'-')
						print(f'{bar} | ---GO LONG---')
					self.go_long(bar-1, amount=self.initialAmount)
					self.position=1				
			elif self.position==1:
				if action==-1:
					if self.verbose==True:
						print(50*'-')
						print(f'{bar} | ---GO SHORT---')
					self.go_short(bar-1, units=self.units)
					self.position=-1			
		self.close_out(bar)

if __name__ == '__main__':
	testLongShort=BacktestLongShort('close', 0, 8000, 10000, verbose=False)
	testLongShort.run_regression_strategy(0, 5000, 5000, 8000, 5)
