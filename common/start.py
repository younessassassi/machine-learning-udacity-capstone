""" Common Functions """

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from matplotlib import style
import numpy as np

DATA_DIR = "data/"
STOCK_DF_DIR = DATA_DIR + "stock_dfs"
TICKERS_PICKLE_DIR = DATA_DIR + "sp500tickers.pickle"
CLASSIFIER_PICKLE_DIR = DATA_DIR + "classifiers/"


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

"""Reads adjusted close stock data for given tickers from CSV files."""
def get_ticker_data(tickers=None, start_date='2006-01-03', end_date='2018-02-23'):
    if not tickers:
        tickers = get_all_tickers()
        tickers.append('SPY')

    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)

    for ticker in tickers:
        df_temp = pd.read_csv(get_ticker_path(ticker), index_col='Date',
                parse_dates=True, usecols=['Date', 'Adj Close'], na_values=['nan'])
        df_temp = df_temp.rename(columns={'Adj Close': ticker})
       
        df = df.join(df_temp)
        if ticker == 'SPY':  # drop dates SPY did not trade
            df.dropna(subset=["SPY"], inplace=True)

        df.fillna(method="ffill",inplace=True) # fill na forward
        df.fillna(method="bfill",inplace=True) # fill na backward
        df.dropna(axis=1, inplace=True) # if stock did not trade at all during specified period then drop it
    return df

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
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    plt.show()

""" Visualize correlation between stocks on a df."""
def visualize_correlation(df):
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
