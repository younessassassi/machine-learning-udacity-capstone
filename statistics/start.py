"""Stock portfolio analysis"""
from statistics.Portfolio import Portfolio
from prediction.Ticker import Ticker, TickerAnalysed
import pandas as pd
import numpy as np
from common.start import get_ticker_data, get_all_tickers, get_prices, get_data_for_symbol
from common.start import visualize_correlation, plot_data, get_tickers_for_symbols, store_ticker_analysis
import scipy.optimize as spo

"""Find optimal allocations for a stock portfolio, optimizing for Sharpe ratio, given the daily prices"""
def find_optimal_allocations(prices):
    guess = 1.0/prices.shape[1]
    function_guess = [guess] * prices.shape[1]
    bnds = [[0,1] for _ in prices.columns]
    cons = ({ 'type': 'eq', 'fun': lambda function_guess: 1.0 - np.sum(function_guess) })
    result = spo.minimize(error_optimal_allocations, function_guess, args = (prices,), method='SLSQP', bounds = bnds, constraints = cons, options={'disp':False})
    allocs = result.x
    return allocs

"""A helper function for the above function to minimize over"""
def error_optimal_allocations(allocs, prices):
    port_val = get_portfolio_value(prices, allocs, 1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_statistics(port_val)
    error = sharpe_ratio * -1
    return error

"""Calculate portfolio statistics"""
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

"""Calculate the portfolio value"""
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

"""Retrieve the list of symbols used"""
def get_symbols_used(stock_prices):
    # some stocks may not have been trading during the period selected.  
    # The dataframe only returns stocks that were atleast partially traded at the time
    return stock_prices.columns.values

"""Retrieve a list of allocations used for each stock in the portfolio"""
def get_allocations_used(symbols_used, allocations):
    allocations_used = allocations[:]
    if len(allocations_used) == 0:
        for symbol in symbols_used:
            allocations_used.append(1/float(len(symbols_used)))
    return allocations_used


"""Retrieve the top 10 stocks selected for the portfolio."""
def get_top_tickers_alloc(symbols_used, allocations_used):
    symbol_alloc = pd.DataFrame({'Tickers': symbols_used, 'Allocations': allocations_used})
    symbol_alloc.set_index('Tickers', inplace=True)
    symbol_alloc.sort_values('Allocations', ascending=False, inplace=True)
    rows, columns = symbol_alloc.shape
    is_reduced = False
    if rows > 10:
        is_reduced = True
        return symbol_alloc.head(10), is_reduced

    return symbol_alloc, is_reduced

"""Find the tickers and correspoding allocations that make up the optimal portfolio"""
def get_top_optimal_symbols(prices):
    allocations_used = find_optimal_allocations(prices)
    allocations_used = allocations_used / np.sum(allocations_used)  # normalize allocations, if they don't sum to 1.0
    symbols_used = get_symbols_used(prices)
    return get_top_tickers_alloc(symbols_used, allocations_used)
    

"""Get the tickers, their correspoding allocations and dataframe that make up the optimal portfolio"""
def optimize_portfolio(portfolio):
    symbol_alloc, is_reduced = get_top_optimal_symbols(portfolio.ticker_prices)
    if is_reduced:
        symbols_used = symbol_alloc.index.values
        symbol_alloc, is_reduced = get_top_optimal_symbols(portfolio.ticker_prices)
    
    symbols_used = symbol_alloc.index.values.tolist()
    allocations_used = symbol_alloc['Allocations'].values.tolist()
    tickers = get_tickers_for_symbols(symbols_used, portfolio.start_date, portfolio.end_date)
    
    return Portfolio(tickers, allocations_used, 
                    portfolio.start_date, portfolio.end_date, portfolio.investment) 
        
"""Compare portfolio to S&P stock index"""
def compare_to_SP(portfolio):
    symbols = ['SPY']
    weights = [1]
    prices_df = get_ticker_data(symbols, portfolio.start_date, portfolio.end_date)
    tickers = [Ticker(symbol=symbols[0], data_df=prices_df)]

    portfolio_sp = Portfolio(tickers, weights, portfolio.start_date, portfolio.end_date, portfolio.investment)
    portfolio.describe()
    print '---------------------------'
    portfolio_sp.describe()
    # Compare daily portfolio value with SPY using a normalized plot
    combined_df = pd.concat([portfolio.value,  portfolio_sp.value], keys=['Portfolio', 'SPY'], axis=1)
    normalized_df = combined_df/combined_df.ix[0]
    plot_data(normalized_df)

"""Plot ticker Adjusted Close prices"""
def display_ticker(symbol, start_date, end_date):
    tickers = get_tickers_for_symbols([symbol], start_date, end_date)
    ticker = tickers[0]
    plot_data(ticker.df, symbol+' Adj Close Price')


""" You can change the stock, allocation, investment and date to get new values. """
def run():
    start_date = '2016-12-23'
    end_date = '2017-01-05'
    investment = 100000 # $100,000.00 as starting investment
    symbols = ['PGR', 'CCI', 'STZ', 'WYNN', 'DPS']
    print 'Data for ' + symbols[0], get_data_for_symbol(symbols[0]).head()

    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    optimize = False
    
    # analyse_portfolio(symbols, weights, start_date, end_date, investment, optimize)    
    
    # print calculated features and label
    prices_df, prices_df_with_spy = get_prices([symbols[0]], start_date, end_date)
    ticker_analysed = TickerAnalysed(symbol=symbols[0], data_df=prices_df[[symbols[0]]])
    print  'Stats for: ', ticker_analysed.symbol
    # Original data
    store_ticker_analysis(ticker_analysed.df, ticker_analysed.symbol+'_original')
    ticker_analysed.print_df()
    # after filling missing data
    store_ticker_analysis(ticker_analysed.get_features_df(), ticker_analysed.symbol+'_fill')
    print ticker_analysed.get_features_df()
    
    tickers = get_tickers_for_symbols(symbols, start_date, end_date)

    portfolio = Portfolio(tickers, weights, start_date, end_date, investment)
    print 'Before Optimization'
    print '---------------------------'
    print 'Comparing portfoltio to S&P'
    print '---------------------------'
    compare_to_SP(portfolio)

    optimized_portfolio = optimize_portfolio(portfolio)
    print '---------------------------'
    print 'After Optimization: '
    print '---------------------------'
    optimized_portfolio.describe()
    print '---------------------------'
    print 'Comparing portfoltio to S&P'
    print '---------------------------'
    compare_to_SP(optimized_portfolio)

if __name__ == "__main__":
    run()
