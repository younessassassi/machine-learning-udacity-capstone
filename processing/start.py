"""Data Processing"""
import bs4 as bs #library for web scraping
import pickle
import requests
import os # create new directories
import time
import pandas as pd
import datetime as dt
import pandas_datareader as pdr


from common.start import plot_data, visualize_correlation, save_sp500_tickers, get_ticker_path, DATA_DIR, STOCK_DF_DIR, TICKERS_PICKLE_DIR


def get_data_from_yahoo(start, end, update_sp500_data=False):
    if start or end:
        # if you change the start or end date, then we need to update the stock data
        update_sp500_data = True

    if not start:
        start = dt.datetime(2006, 1, 1) 
    if not end:
        end = dt.datetime.now() - dt.timedelta(days=1)

    if not os.path.exists(STOCK_DF_DIR):
        os.makedirs(STOCK_DF_DIR)

    
    if update_sp500_data:
        tickers = save_sp500_tickers()
        dir_files = os.listdir(STOCK_DF_DIR)
        for item in dir_files:
            if item.endswith(".csv"):
                os.remove(os.path.join(STOCK_DF_DIR, item))
        
        counter = 0
        
        # add S&P 500 index
        tickers.append("SPY")

        for ticker in tickers:
            counter += 1
            if (counter % 25 == 0):
                print "{} stocks finished updating.".format(counter)
            if not os.path.exists(get_ticker_path(ticker)):
                df = pdr.get_data_yahoo(ticker.replace('.', '-'), start, end)
                df.to_csv(get_ticker_path(ticker))
                time.sleep(.100) # I added this timeout to ensure that yahoo doesnt cancel my transaction requests 
            else:
                print('Already have {}'.format(ticker))
    

# def create_sp500_joined_closes_file():
#     if os.path.exists(sp500_joined_closes):
#         os.remove(sp500_joined_closes)

#     with open(tickers_pickle, "rb") as f:
#         tickers = pickle.load(f)
        
#     main_df = pd.DataFrame()
    
#     for count,ticker in enumerate(tickers):
#         df = pd.read_csv(get_ticker_path(ticker))
#         df.set_index('Date', inplace=True)
        
#         df.rename(columns = {'Adj Close':ticker}, inplace=True)
#         df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
        
#         if main_df.empty:
#             main_df = df
#         else:
#             main_df = main_df.join(df, how='outer')
            
#         if count % 10 == 0:
#             print count

    # main_df.to_csv(sp500_joined_closes)

# def createReducedDataFrame():
#     startDate = "2012-01-01" 
#     endDate = "2014-01-01"
#     df = pd.read_csv(sp500_joined_closes)
#     df.set_index('Date', inplace=True)
#     newDF = df[startDate: endDate]
#     newDF.fillna(method="ffill",inplace=True)
#     newDF.fillna(method="bfill",inplace=True)
#     newDF.to_csv(sp500_reduced_joined_closes)

#     spDf = pd.read_csv(sp_index)
#     spDf.set_index('Date', inplace=True)
#     newSpDf = spDf[startDate: endDate]
#     newSpDf.fillna(method="ffill",inplace=True)
#     newSpDf.fillna(method="bfill",inplace=True)
#     newSpDf.to_csv(sp500_index_reduced)
#     # visualize_correlation(sp500_reduced_joined_closes)

"""Retrieve S&P data"""
def run():
    update_stock_prices = True # set to True to update the stock prices
    start_time = None # set to True to update the stock prices
    end_time = None
    if update_stock_prices or start_time or end_time:   
        get_data_from_yahoo(start_time, end_time, update_stock_prices)


if __name__ == "__main__":
    run()