import pandas as pd
import numpy as np

"""Trade simulation class that can be initialized with a portfolio object, and a date range"""
class Sim(object):
    
    def __init__(self, portfolio, buy_date, sell_date):
        self.portfolio = portfolio
        self.buy_date = buy_date
        self.sell_date = sell_date
        #set up the data frame for the stocks over the period
        self.daily_prices = portfolio._get_ticker_prices()
        
        self.trades_df = self.daily_prices.copy()
        self.trades_df[0:] = 0.0
        self.trades_df['Residual'] = 0.0
        self.prepare_trades()
        pass 

    """Method for calculating the cash value after liquidating the portolio positions plus any unused cash"""
    def cash_out(self):
        assets = self.trades_df.loc[self.buy_date]
        residual = assets[-1]
        assets = assets[:-1]
        sell_data_prices = self.daily_prices.loc[self.sell_date]
        calculated_assets = np.multiply(assets, sell_data_prices)
        cash_total = np.sum(calculated_assets) + residual
        return cash_total

    """Calculate the number of stocks that can be purchased based on available cash and portfolio weight for a single stock"""
    def get_trade_count(self, price, weight, investment):
        # number of shares you can purchase
        count = (investment * weight) / price
        residual = (investment * weight) % price
        return int(count), residual
        
    """Prepare the trade information for all stocks making up the portfolio"""
    def prepare_trades(self):
        #step through the trades data frame
        cash_total = self.portfolio.investment
        for index, row in self.trades_df.iterrows():
            residual = 0
            for symbol in self.portfolio.symbols:
                count, temp_res = self.get_trade_count(self.daily_prices.loc[index, symbol], self.portfolio.weight_dict[symbol], cash_total)
                residual = residual + temp_res
                row[symbol] = count
            row['Residual'] = residual
        pass

    """Return a DataFrame with the list of trades"""
    def get_trades_df(self):
        return self.trades_df

    """Return the daily prices"""
    def get_original_prices(self):
        return self.daily_prices

    """Get the daily portfolio daily retruns"""
    def get_daily_returns(self):
        return self.portfolio.daily_returns
    
    """Get the portfolio estimated value"""
    def get_value(self):
        return self.portfolio.value