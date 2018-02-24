"""Stock portfolio analysis"""
import pandas as pd
import numpy as np
from common.start import get_ticker_data, get_all_tickers
from common.start import visualize_correlation, plot_data
import scipy.optimize as spo

#################################################################################
"""Find optimal allocations for a stock portfolio, optimizing for Sharpe ratio, given the daily prices"""
def find_optimal_allocations(prices):
    guess = 1.0/prices.shape[1]
    function_guess = [guess] * prices.shape[1]
    bnds = [[0,1] for _ in prices.columns]
    cons = ({ 'type': 'eq', 'fun': lambda function_guess: 1.0 - np.sum(function_guess) })
    result = spo.minimize(error_optimal_allocations, function_guess, args = (prices,), method='SLSQP', bounds = bnds, constraints = cons, options={'disp':True})
    allocs = result.x
    return allocs

"""A helper function for the above function to minimize over"""
def error_optimal_allocations(allocs, prices):
    port_val = get_portfolio_value(prices, allocs, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)
    error = sharpe_ratio * -1
    return error


#################################################################################
def get_portfolio_statistics(portfolio_value, daily_rf=0, samples_per_year=252):
    cummulative_return = (portfolio_value[-1]/portfolio_value[0])-1

    daily_returns = portfolio_value.copy()
    daily_returns = (daily_returns/daily_returns.shift(1)) - 1
    daily_returns.ix[0, 0] = 0 #set the first number to 0

    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    #sharpe ratio - average return earned in excess of the risk-free rate per unit of volatility or total risk
    sharpe_ratio = ((daily_returns - daily_rf).mean()/daily_returns.std()) * np.sqrt(samples_per_year)

    return cummulative_return, avg_daily_return, std_daily_return, sharpe_ratio

def get_portfolio_value(stock_prices, allocations, starting_investment):
    #normalize the stock prices
    df = stock_prices / stock_prices.ix[0] 
    # normalized values times their corresponding allocation
    df = df * allocations
    #allocated value times the starting investment
    df = df * starting_investment
    #determine the entire portfolio value on each day by summing all of the columns. 
    portfolio_value = df.sum(axis=1)
    
    return portfolio_value

def get_alloc_tickers_used(df, allocations):
    allocations_used = allocations[:]
    tickers_used = df.columns.values
    if len(allocations_used) == 0:
        for ticker in tickers_used:
            allocations_used.append(1/float(len(tickers_used)))
    print 'Number of tickers used: ', len(allocations_used)
    return tickers_used, allocations_used


"""Compare performance with the S&P500 for the same period"""
def analyse_portfolio(tickers, allocations, start_date, end_date, starting_investment):
    tickers_with_SPY = tickers[:]
    tickers_with_SPY.append('SPY')
    prices_with_SPY = get_ticker_data(tickers_with_SPY, start_date, end_date)
    prices_without_SPY = prices_with_SPY.drop('SPY', axis=1)

    tickers_used, allocations_used = get_alloc_tickers_used(prices_without_SPY, allocations)
    # allocations_used = find_optimal_allocations(prices_without_SPY)

    # Get daily portfolio value
    portfolio_values = get_portfolio_value(prices_without_SPY, allocations_used, starting_investment)
   
    # Get portfolio statistics
    cummulative_return, avg_daily_return, std_daily_return, sharpe_ratio = get_portfolio_statistics(portfolio_values)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Tickers:", tickers_used
    print "Allocations used:", allocations_used
    print "Sharpe Ratio:", sharpe_ratio
    print "Volatility:", std_daily_return
    print "Average Daily Return:", avg_daily_return
    print "Cumulative Return:", cummulative_return

    # visualize_correlation(df)
    # Compare daily portfolio value with SPY using a normalized plot
    combined_df = pd.concat([portfolio_values,  prices_with_SPY['SPY']], keys=['Portfolio', 'SPY'], axis=1)
    normalized_df = combined_df/combined_df.ix[0]
    plot_data(normalized_df)
  

"""Though this class is mostly used for helper functions, you can execute all of them here for testing.
Some default options are provided, though these can be changed to the stock, allocation, and date of your choice. """
def run():
    start_date = '2018-02-01'
    end_date = '2018-02-03'
    starting_investment = 100000 # $100,000.00 as starting investment
   
    tickers = get_all_tickers()
    allocations = []
    # allocations = [0.5, 0.5] # allocations must add up to 1
    
    analyse_portfolio(tickers, allocations, start_date, end_date, starting_investment)


if __name__ == "__main__":
    run()
