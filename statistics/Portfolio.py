from prediction.Ticker import Ticker

import numpy as np
import pandas as pd

class Portfolio(object):
    def __init__(self, tickers, weights, investment=1, optimize=False,
        daily_rf=0, samples_per_year=252):
        self.daily_rf = daily_rf
        self.samples_per_year = samples_per_year
        self.tickers = tickers
        self.weights = weights
        self.investment = investment
        self.optimize = optimize
        self.ticker_prices = self._get_ticker_prices()
        self.value = self.get_value()

        self.cummulative_return = (self.value[-1]/self.value[0])-1

        self.daily_returns = self.value.copy()
        self.daily_returns = (self.daily_returns/self.daily_returns.shift(1)) - 1
        self.daily_returns.ix[0, 0] = 0 #set the first number to 0

        self.avg_daily_return = self.daily_returns.mean()
        self.std_daily_return = self.daily_returns.std()

        #sharpe ratio - average return earned in excess of the risk-free rate per unit of volatility or total risk
        self.sharpe_ratio = ((self.daily_returns - self.daily_rf).mean()/self.daily_returns.std()) * np.sqrt(self.samples_per_year)

        pass

    def describe(self):
        print 'insvestment: ', self.investment

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


    # def get_optimized_portfolio_params(self):

    #     return o

    # def calculate_stats(self):
        
    #     tickers_with_SPY = tickers[:]
    #     tickers_with_SPY.append('SPY')
    #     prices_with_SPY = get_ticker_data(tickers_with_SPY, start_date, end_date)
    #     prices_without_SPY = prices_with_SPY.drop('SPY', axis=1)
        
    #     if optimize:
    #     prices_without_SPY, tickers_used, allocations_used = get_optimized_portfolio_params(prices_without_SPY)
    #     else:
    #         tickers_used = get_tickers_used(prices_without_SPY)
    #         allocations_used = get_allocations_used(tickers_used, allocations)

        
    #     # Get daily portfolio value
    #     portfolio_values = get_portfolio_value(prices_without_SPY, allocations_used, starting_investment)
    
    #     # Get portfolio statistics
    #     cummulative_return, avg_daily_return, std_daily_return, sharpe_ratio = get_portfolio_statistics(portfolio_values)

    #     # Print statistics
    #     print "Start Date:", start_date
    #     print "End Date:", end_date
    #     print "Tickers:", tickers_used
    #     print "Allocations used:", allocations_used
    #     print "Sharpe Ratio:", sharpe_ratio
    #     print "Volatility:", std_daily_return
    #     print "Average Daily Return:", avg_daily_return
    #     print "Cumulative Return:", cummulative_return

    #     # Compare daily portfolio value with SPY using a normalized plot
    #     combined_df = pd.concat([portfolio_values,  prices_with_SPY['SPY']], keys=['Portfolio', 'SPY'], axis=1)
    #     normalized_df = combined_df/combined_df.ix[0]
    #     plot_data(normalized_df)
    #     return cummulative_return, avg_daily_return, std_daily_return, sharpe_ratio

    

