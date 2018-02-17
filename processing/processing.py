"""Data Processing"""
import bs4 as bs #library for web scraping
import pickle
import requests
import os # create new directories
import time

import datetime as dt
import pandas_datareader as pdr

data_directory = "data/"

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open(data_directory + "sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers

def get_data_from_yahoo(start, end, reload_sp500_ticker=False, update_sp500_data=False):
    stock_dfs_directory = data_directory + "stock_dfs";
    if start or end:
        # if you change the start or end date, then we need to update the stock data
        update_sp500_data = True
    if not start:
        start = dt.datetime(2000, 1, 1) 
    if not end:
        end = dt.datetime.now() - dt.timedelta(days=1)

    if not os.path.exists(stock_dfs_directory):
        os.makedirs(stock_dfs_directory)
    
    if update_sp500_data:
        dir_files = os.listdir(stock_dfs_directory)
        for item in dir_files:
            if item.endswith(".csv"):
                os.remove(os.path.join( stock_dfs_directory, item))

    if reload_sp500_ticker:
        tickers = save_sp500_tickers()
    else:
        with open(data_directory +"sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    
    counter = 0
    for ticker in tickers:
        counter += 1
        if (counter % 25 == 0):
            print "{} stocks finished updating.".format(counter)
        if not os.path.exists(stock_dfs_directory+'/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker.replace('.', '-'), start, end)
            df.to_csv(stock_dfs_directory+'/{}.csv'.format(ticker))
            time.sleep(.300) # I added this timeout to ensure that yahoo doesnt cancel my transaction requests 
        else:
           print('Already have {}'.format(ticker))

"""Retrieve S&P data"""
def run():
    save_sp500_tickers()
    update_stock_tickers = True # refreshes the stock tickers used to retrieve stock prices
    update_stock_prices = True # updates the stock prices from  up to yesterday
    start_time = None 
    end_time = None
    get_data_from_yahoo(start_time, end_time, update_stock_tickers, update_stock_prices)


if __name__ == "__main__":
    run()