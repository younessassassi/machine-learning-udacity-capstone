from prediction.Ticker import Ticker

import numpy as np

"""Portfolio class that can be initialized using tickers objects, weights, date range and investments """
class Portfolio(object):
    def __init__(self, tickers, weights, start_date, end_date, 
                investment=1, daily_rf=0, samples_per_year=252):
        
        self.start_date = start_date
        self.end_date = end_date
        self.daily_rf = daily_rf
        self.samples_per_year = samples_per_year
        self.tickers = tickers
        self.weights = weights
        self.investment = investment
        self.ticker_prices = self._get_ticker_prices()
        self.value = self.get_value()
        self.symbols = [ticker.symbol for ticker in tickers]
        
        self.cummulative_return = (self.value[-1]/self.value[0])-1

        self.daily_returns = self.value.copy()
        self.daily_returns = (self.daily_returns/self.daily_returns.shift(1)) - 1
        self.daily_returns.ix[0, 0] = 0 #set the first number to 0

        self.avg_daily_return = self.daily_returns.mean()
        self.volatility = self.daily_returns.std()

        #sharpe ratio - average return earned in excess of the risk-free rate per unit of volatility or total risk
        self.sharpe_ratio = ((self.daily_returns - self.daily_rf).mean()/self.daily_returns.std()) * np.sqrt(self.samples_per_year)
        self.weight_dict = dict(zip(self.symbols, self.weights))
        pass


    """ Print important object information """
    def describe(self):
        print 'Portfolio stats From {} to {}'.format(self.start_date, self.end_date)
        print 'Ticker Symbols: ', self.symbols
        print 'Corresponding Weights: ', self.weights
        print 'Sharpe Ratio: ', self.sharpe_ratio
        print 'Volatility: ', self.volatility
        print 'Average Daily Return: ', self.avg_daily_return
        print 'Starting Investment: ${:,.2f}'.format(self.investment)
        print 'Ending portfolio value: ${:,.2f}'.format(self.value[-1])
        print 'Cummulative Return {0:.2f}%'.format(self.cummulative_return*100)

    """ returns a dataframe with adj close price of all tickers in the portfolio"""
    def _get_ticker_prices(self):
        if not self.tickers:
            return {}
        ticker = self.tickers[0]
        ticker_prices = self._get_stock_prices(ticker)
        ticker_prices.rename(columns={'Adj Close': ticker.symbol}, inplace=True)
        for ticker in self.tickers[1:]:
            ticker_prices = ticker_prices.join(self._get_stock_prices(ticker))
            ticker_prices.rename(columns={ 'Adj Close': ticker.symbol}, inplace=True)

        return ticker_prices

    """ returns a copy of the ticker adj close price dataframe"""
    def _get_stock_prices(self, ticker):
        return ticker.get_adj_close_df().copy()

    """Calculate the portfolio value"""
    def get_value(self):
        #normalize the stock prices
        df = self.ticker_prices / self.ticker_prices.ix[0] 
        # normalized values times their corresponding allocation
        df = df * self.weights
        #allocated value times the starting investment
        df = df * self.investment
        #determine the entire portfolio value on each day by summing all of the columns. 
        return df.sum(axis=1)