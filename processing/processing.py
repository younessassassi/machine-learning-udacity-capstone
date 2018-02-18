"""Data Processing"""
import bs4 as bs #library for web scraping
import pickle
import requests
import os # create new directories
import time
import pandas as pd
import datetime as dt
import pandas_datareader as pdr


import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

from common.common import plot_data

data_directory = "data/"
stock_dfs_directory = data_directory + "stock_dfs"
tickers_pickle = data_directory + "sp500tickers.pickle"
sp500_joined_closes = data_directory+'sp500_joined_closes.csv'
sp500_reduced_joined_closes = data_directory+'sp500_reduced_joined_closes.csv'
sp500_clean_joined_closes = data_directory+'sp500_clean_joined_closes.csv'

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open(tickers_pickle, "wb") as f:
        pickle.dump(tickers, f)
    return tickers

def getTickerPath(ticker):
    return stock_dfs_directory+'/{}.csv'.format(ticker)

def get_data_from_yahoo(start, end, reload_sp500_ticker=False, update_sp500_data=False):
    if start or end:
        # if you change the start or end date, then we need to update the stock data
        update_sp500_data = True
    if not start:
        start = dt.datetime(2000, 1, 1) 
    if not end:
        end = dt.datetime.now() - dt.timedelta(days=1)

    if not os.path.exists(stock_dfs_directory):
        os.makedirs(stock_dfs_directory)
    
    if reload_sp500_ticker:
        tickers = save_sp500_tickers()
    else:
        with open(tickers_pickle, "rb") as f:
            tickers = pickle.load(f)
    
    if update_sp500_data:
        dir_files = os.listdir(stock_dfs_directory)
        for item in dir_files:
            if item.endswith(".csv"):
                os.remove(os.path.join(stock_dfs_directory, item))
        
        counter = 0
        for ticker in tickers:
            counter += 1
            if (counter % 25 == 0):
                print "{} stocks finished updating.".format(counter)
            if not os.path.exists(getTickerPath(ticker)):
                df = pdr.get_data_yahoo(ticker.replace('.', '-'), start, end)
                df.to_csv(getTickerPath(ticker))
                time.sleep(.300) # I added this timeout to ensure that yahoo doesnt cancel my transaction requests 
            else:
                print('Already have {}'.format(ticker))


def create_sp500_joined_closes_file():
    if os.path.exists(sp500_joined_closes):
        os.remove(sp500_joined_closes)

    with open(tickers_pickle, "rb") as f:
        tickers = pickle.load(f)
        
    main_df = pd.DataFrame()
    
    for count,ticker in enumerate(tickers):
        df = pd.read_csv(getTickerPath(ticker))
        df.set_index('Date', inplace=True)
        
        df.rename(columns = {'Adj Close':ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
            
        if count % 10 == 0:
            print count

    main_df.to_csv(sp500_joined_closes)

def visualize_data():
    df = pd.read_csv(sp500_joined_closes)
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

def cleanup():
    print 'cleanup'
    df = pd.read_csv(sp500_reduced_joined_closes)
    df.set_index('Date', inplace=True)
    plot_data(df)
    
    # hm_days = 7
    # df.pd.read_csv(sp500_joined_closes, index_col=0)
    # tickers = df.columns.values.tolist()
    # df.fillna(0, inplace=True)
    
    # for i in range(1, hm_days+1):
    #     df['{}_{}'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    # df.fillna(0, inplace=True)
    # df.to_csv(clean_sp500_joined_closes)

def createReducedDataFrame():
    startDate = "2016-01-01" 
    endDate = "2018-01-01"
    df = pd.read_csv(sp500_joined_closes)
    df.set_index('Date', inplace=True)

    newDF = df[startDate: endDate]
    newDF.to_csv(sp500_clean_joined_closes)

"""Retrieve S&P data"""
def run():
    update_stock_tickers = False # set to True to refresh the stock tickers used to retrieve stock prices
    update_stock_prices = False # set to True to update the stock prices
    start_time = None # set to True to update the stock prices
    end_time = None
    if update_stock_tickers or update_stock_prices or start_time or end_time:   
        get_data_from_yahoo(start_time, end_time, update_stock_tickers, update_stock_prices)
        create_sp500_joined_closes_file()
        createReducedDataFrame()

    cleanup()
    # visualize_data()


if __name__ == "__main__":
    run()