import numpy as np
import pandas as pd

class Ticker(object):
    def __init__(self, symbol=None, stocks={},
                 start_date=None, end_date=None, volatility_window=5,
                 momentum_window=5, bollinger_bands_window=5, moving_average_window=5):
        self.symbol=symbol
        self.max_window=max(momentum_window, bollinger_bands_window, moving_average_window)
        self.momentum_window=momentum_window
        self.moving_average_window=moving_average_window
        self.bollinger_bands_window=bollinger_bands_window
        
        dates = pd.date_range(start_date, end_date)

        stock = stocks[symbol]
        features_df = pd.DataFrame(index=dates)

        features_df = features_df.join(stock)
 
        features_df.rename(columns={symbol: 'Adj Close'}, inplace=True)
 
        features_df.fillna(method='ffill', inplace=True)
        features_df.fillna(method='bfill', inplace=True)
        features_df.dropna()
        
        self.original_df=features_df.copy()
        
        features_df=features_df.join(self._calculate_daily_returns(df=self.original_df))
        features_df=features_df.join(self._calculate_cumulative_returns(df=self.original_df))
        features_df=features_df.join(self._calculate_momentum(df=self.original_df))
        features_df=features_df.join(self._calculate_moving_average(df=self.original_df))
        bb_upper_bound,bb_lower_bound=self._calculate_bb(df=self.original_df)
        features_df=features_df.join(bb_upper_bound)
        features_df=features_df.join(bb_lower_bound)
        features_df=features_df.join(self._add_label_price(df=self.original_df))
        
        self.df=features_df
        
        pass
    
    def _get_clean_df(self):
        df = self.df.copy()
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.dropna()
      
        return df

    def get_features(self):
        features = self._get_clean_df()
        features.drop(['Adj Close', 'Label'], axis=1 , inplace=True)
        return features.values
    
    def get_label(self):
        df = self._get_clean_df()
        label = df['Label']
        return label.values
    
    
    def _add_label_price(self, df=None):
        return (df.shift(-1)).rename(columns={'Adj Close': 'Label'})

    
    def print_df(self):
        print self.df
    
    def _calculate_daily_returns(self, df=None):
        daily_returns=df.copy()
        daily_returns=(daily_returns/daily_returns.shift(1))-1
        return daily_returns[1:].rename(columns={'Adj Close': 'Daily Returns'})
    
    def _calculate_cumulative_returns(self, df=None):
        return ((df/df.ix[0]) - 1).rename(columns={'Adj Close': 'Cumulative Returns'})

    def _calculate_momentum(self, df=None):
        n=self.momentum_window
        
        df=df/df.shift(n)
        return (df[n:] - 1).rename(columns={'Adj Close': 'Momentum'})

    def _calculate_volatility(self, df=None):
        n=self.volatility_window
        return ((df.rolling(window=n,center=False).std())[n:] - 1 ).rename(columns={'Adj Close': 'Volatility'})
   
    def _calculate_moving_average(self, df=None):
        n=self.moving_average_window
        
        return ((df/df.rolling(window=n,center=False).mean())[n:] - 1).rename(columns={'Adj Close': 'Moving Average'})

    def _calculate_bb(self, df=None):
        n=self.bollinger_bands_window
        
        std=(df.rolling(window=n,center=False).std())[n:]
        ma=(df.rolling(window=n,center=False).mean())[n:]
        upper_bound=ma+2*std
        lower_bound=ma-2*std
        return upper_bound.rename(columns={'Adj Close': 'BB Upper Bound'}), lower_bound.rename(columns={'Adj Close': 'BB Lower Bound'})

    def _normalize(self, df):
        return (df - df.mean())/df.std()