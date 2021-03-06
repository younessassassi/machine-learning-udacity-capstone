""" Common Functions """

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from prediction.Ticker import Ticker
from matplotlib import style
import numpy as np

DATA_DIR = "data/"
STOCK_DF_DIR = DATA_DIR + "stock_dfs"
TICKERS_PICKLE_DIR = DATA_DIR + "sp500tickers.pickle"
CLASSIFIER_PICKLE_DIR = DATA_DIR + "classifiers/"
TICKER_STATS_DIR = DATA_DIR + "ticker_stats/"

"""Retrieve the list of S&P 500 tickers"""
def get_all_tickers():
    tickers = False
    with open(TICKERS_PICKLE_DIR, "rb") as f:
        tickers = pickle.load(f)
    
    if not tickers:
        tickers = save_sp500_tickers()
    
    return tickers

"""return ticker path"""
def get_ticker_path(ticker):
    return STOCK_DF_DIR + '/{}.csv'.format(ticker)

"""Returns two dataframes that contain adj close for each of the symbols provided plus the S&P index"""
def get_prices(symbols, start_date, end_date, no_spy=True):
    if no_spy:
        symbols_with_SPY = symbols[:]
        symbols_with_SPY.append('SPY')
        prices_with_SPY = get_ticker_data(symbols_with_SPY, start_date, end_date)
        prices_without_SPY = prices_with_SPY.drop('SPY', axis=1)
    else:
        prices = get_ticker_data(symbols, start_date, end_date)
        prices_with_SPY = prices.copy()
        prices_without_SPY = prices.copy()
    return prices_without_SPY, prices_with_SPY

"""Return a DataFrame with original ticker data downloaded from the web"""
def get_data_for_symbol(symbol):
    ticker_data = pd.read_csv(get_ticker_path(symbol), index_col='Date',
                parse_dates=True,  na_values=np.nan)
    return ticker_data

"""Reads adjusted close stock data for given tickers from CSV files."""
def get_ticker_data(tickers=None, start_date='2006-01-03', end_date='2018-02-23'):
    if not tickers:
        tickers = get_all_tickers()
        tickers.append('SPY')

    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    for ticker in tickers:
        df_temp = pd.read_csv(get_ticker_path(ticker), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=np.nan)
       
        df_temp = df_temp.rename(columns={'Adj Close': ticker})
        df = df.join(df_temp)
        if ticker == 'SPY':  # drop dates SPY did not trade
            df.dropna(subset=["SPY"], inplace=True)

    return df

"""Get a list of Ticker objects using the symbols and date range"""
def get_tickers_for_symbols(symbols, start_date, end_date, no_spy=True):
    prices_df, prices_df_with_spy = get_prices(symbols, start_date, end_date, no_spy)
    tickers = None
    for symbol in symbols:
        ticker = Ticker(symbol=symbol, data_df=prices_df[[symbol]])
        if tickers == None:
            tickers = [ticker]
        else: 
            tickers.append(ticker)

    return tickers

"""Retrieve the list tickers that make up the S&P index"""
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open(TICKERS_PICKLE_DIR, "wb") as f:
        pickle.dump(tickers, f)
    return tickers

"""Plot stock prices with a custom title and axis labels."""
def plot_data(df, title="Stock performance", xlabel="Date", ylabel="Price"):
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend(loc=2)
    plt.show()

""" Visualize correlation between stocks on a dataframe."""
def visualize_correlation(df, title=""):
    df_corr = df.corr()
    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    column_labels = df_corr.columns
    row_labels = df_corr.index
    
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    plt.title = title
    heatmap.set_clim(-1, 1)
    plt.tight_layout
    
    plt.show()


""" get data from pickle """
def get_pickle(filename, path):
    if not os.path.exists(path+filename):
        print 'file does not exist in path ' + path
        return
    
    with open(path+filename, "rb") as f:
        data_pickle = pickle.load(f)

    return data_pickle

""" store data using pickle """
def store_pickle(data, filename, path):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path+filename, "wb") as f:
        pickle.dump(data, f)

"""Store the optimized SP500 portfolio"""
def store_optimized_portfolio(portfolio):
    store_pickle(portfolio, 'optimized_portfolio.pickle', DATA_DIR)

"""Get the optimized SP500 portfolio"""
def get_optimized_portfolio():
    return get_pickle('optimized_portfolio.pickle', DATA_DIR)

"""Store the model analysis data in a csv file"""
def store_classifer_analysis(df):
    if not os.path.exists(CLASSIFIER_PICKLE_DIR):
        os.makedirs(CLASSIFIER_PICKLE_DIR)
    
    df.to_csv(TICKER_STATS_DIR+'classifiers_results.csv')

"""Get stored model analysis data"""
def get_classifier_analysis():
    if not os.path.exists(CLASSIFIER_PICKLE_DIR):
        print 'Classifier stat file does not exit. Please run the predictions before you try openning this files'
    else:
        return pd.read_csv(TICKER_STATS_DIR+'classifiers_results.csv')

"""Store the stock analysis data in a csv file"""
def store_ticker_analysis(df, symbol):
    if not os.path.exists(TICKER_STATS_DIR):
        os.makedirs(TICKER_STATS_DIR)
    
    df.to_csv(TICKER_STATS_DIR+symbol+'.csv')
