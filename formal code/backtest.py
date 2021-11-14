import numpy as np
import pandas as pd
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

'''
Base class for the event-based testing of the trading strategies

Attributes
------------------------------
column : column to be used for trading
start  : starting date for the collection of data for trading
end    : ending date for the collection of data for trading
amount : amount to be traded 
ftc    : fixed transaction costs
ptc    : proportional transaction costs

Functions
-----------------------------------
get_data()         : retrieves the data for implementing the trading strategy
plot_data()        : plots the retrieved data
get_price()        : returns the price for the corresponding bar
print_balance()    : prints the current balance left
print_net_wealth() : prints the current net wealth
place_buy_order()  : places an order to buy foreign exchange
place_sell_order() : places an order to sell foreign exchange
close_out()        : closes the implementation of the trade
'''
class BacktestBase(object):
	def __init__(self, column, start, end, amount, ftc=0.0, ptc=0.0, verbose=True):
		self.column=column
		self.start=start
		self.end=end
		self.initialAmount=amount    #stores the inital ammount allotted for trading (remains constant)
		self.amount=amount           #trading cash balance (running balance)
		self.ftc=ftc                 #fixed transaction costs per trade
		self.ptc=ptc                 #propertional transaction costs per trade
		self.units=0                 #number of units traded
		self.position=0              #initial position is set to neutral
		self.trades=0                #no trades are executed initially
		self.verbose=verbose         #prints detailed summary in terminal
		self.get_data()
	'''
	Retrieve and prepare the data in the base class
	'''
	def get_data(self):
		'''
		Retrieve and prepare the features
		'''
		raw=pd.read_csv('./OneMonthData.csv', index_col=0, parse_dates=True).dropna() #read the data
		raw=pd.DataFrame(raw[self.column][self.start:self.end])
		raw['returns']=np.log(raw/raw.shift(1))
		self.data=raw.dropna()
	'''
	Define some helper functions
	'''	
	def plot_data(self, cols=None):
		'''
		Plot the requested column of the time series
		'''
		if cols is None:
			cols=['close']
		self.data[cols].plot(figsize=(10, 6), title=cols[0])
		
	def get_price(self, bar):
		'''
		For the requested bar, returns the price
		'''
		price=self.data[self.column][bar]
		return bar, price
		
	def set_prices(self, price):
		'''
		Keep track of prices to track the performance
		'''
		self.entry_price=price
		self.min_price=price
		self.max_price=price
		
	def print_balance(self, bar):
		'''
		For the requested bar, prints the available balance
		'''
		date, price=self.get_price(bar)
		print(f'{date} | current balance {self.amount:.2f}')

	def print_net_wealth(self, bar):
		'''
		For the requested bar, prints the net wealth
		'''
		date, price=self.get_price(bar)  
		net_wealth=self.units*price+self.amount
		print(f'{date} | current net wealth {net_wealth:.2f}')
	
	'''
	Define function to place orders
	'''	
	def place_buy_order(self, bar, units=None, amount=None):
		'''
		Place an order to buy foreign exchange for the passed amount or the passed number of units at the corresponding bar
		'''
		date, price=self.get_price(bar)  #retrieve the price for the requested tick value
		if units is None:                #if the number of unites are not given
			units=int(amount/price)      #number of units are assumed to be all units
		self.amount-=(units*price)*(1+self.ptc)+self.ftc  #update the current balance
		self.units+=units                #the number of units are updated
		self.trades+=1                   #the number of trades is increased by one
		self.set_prices(price)           #the prices are updated
		if self.verbose:
			print(f'{date} | buying {units} units at {price:.3f}')
			self.print_balance(bar)        #current banace is printed
			self.print_net_wealth(bar)     #current wealth is printed
			
	def place_sell_order(self, bar, units=None, amount=None):
		'''
		Place an order to sell foreign exchange for the passed amount or the passed number of units at the corresponding bar
		'''
		date, price=self.get_price(bar)  #retrieve the price for the requested tick value
		if units is None:                #if the number of unites are not given
			units=int(amount/price)      #number of units are assumed to be all units
		self.amount+=(units*price)*(1-self.ptc)-self.ftc  #update the current balance
		self.units-=units                #the number of units are updated
		self.trades+=1                   #the number of trades is increased by one
		self.set_prices(price)           #the prices are updated
		if self.verbose:
			print(f'{date} | selling {units} units at {price:.3f}')
			self.print_balance(bar)         #current banace is printed
			self.print_net_wealth(bar)      #current wealth is printed
	
	'''
	Define a function to close out the market position at the end
	'''		
	def close_out(self, bar):
		'''
		Closes out any remaining open position at the corresponding bar 
		'''
		date, price=self.get_price(bar)  #retrieve the price for the requested tick value
		#self.amount+=self.units*price
		if self.units<0:                 
			self.place_buy_order(bar, units=-self.units)
		else:
			self.place_sell_order(bar, units=self.units)
		print(50*'_')
		print(f'{date} | --- CLOSING OUT ---')
		if self.verbose:
			print(f'{date} | inventory {self.units} units at {price:.3f}')
			print(50*'-')
		print('Final balance [$] {:.3f}'.format(self.amount))  #print final balance as the cash balance plus the price of assets
		perf = ((self.amount-self.initialAmount)/self.initialAmount*100)
		print('Net Performance [%] {:.3f}'.format(perf))       #calculate the net performance
		print('Trades Executed [#] {:.3f}'.format(self.trades))
		print(50*'_')
		
if __name__ == '__main__':
	test=BacktestBase('close', 0, 5000, 1000, verbose=False)
	print(test.data.info())
	print(test.data.tail())
	print(test.initialAmount)
	bar=1000
	print(test.get_price(bar))
	test.place_buy_order(bar, amount=100)
	test.print_net_wealth(2*bar)
	test.place_sell_order(2*bar, units=100)
	test.close_out(4*bar)
	test.plot_data()
	plt.show()
