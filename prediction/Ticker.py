import numpy as np

"""Parent class responsible for initializing the Ticker class with basic pricing data """
class Ticker(object):
    def __init__(self, symbol=None, data_df={}):
        self.symbol=symbol
        self.df = data_df.rename(columns={symbol: 'Adj Close'})

        pass
    
    """Fill or drop missing data"""
    def _get_clean_df(self):
        df = self.df.copy()
        df.replace('', np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.dropna(inplace=True)
        return df

    """Return a DataFrame with Ticker adjusted close prices"""
    def get_adj_close_df(self):
        df = self._get_clean_df()
        return df[['Adj Close']]

"""Child class with methohds for creating a set of features for a ticker"""
class TickerAnalysed(Ticker):
    def __init__(self, symbol=None, data_df={}, window=5):
        Ticker.__init__(self, symbol, data_df)
        self.window = window
        self.momentum_window=window
        self.moving_average_window=window
        self.bollinger_bands_window=window
        
        features_df = data_df.copy()
 
        features_df.rename(columns={symbol: 'Adj Close'}, inplace=True)
 
        features_df.fillna(method='ffill', inplace=True)
        features_df.fillna(method='bfill', inplace=True)
        features_df.dropna(inplace=True)
        
        self.original_df=features_df.copy()
        features_df=features_df.join(self._calculate_daily_returns(df=self.original_df))
        features_df=features_df.join(self._calculate_momentum(df=self.original_df))
        bb_upper_bound,bb_lower_bound=self._calculate_bb(df=self.original_df)
        features_df=features_df.join(bb_upper_bound)
        features_df=features_df.join(bb_lower_bound)
        features_df=features_df.join(self._calculate_moving_average(df=self.original_df))
        features_df=features_df.join(self._add_next_day_price(df=self.original_df))
        self.df=features_df
        
        pass
    
    """Returns a set of ticker features that can be used for training a model"""
    def get_features(self):
        features = self._get_clean_df()
        features.drop(['Adj Close', 'Next Day Adj Close'], axis=1 , inplace=True)
       
        # features.drop(['Adj Close', 'Next Day Adj Close', 'Moving Average', 'Momentum'], axis=1 , inplace=True)
        return features.values

    """Returns a set of ticker features as a DataFrame"""
    def get_features_df(self):
        features = self._get_clean_df()
        features.drop(['Adj Close', 'Next Day Adj Close'], axis=1 , inplace=True)
       
        # features.drop(['Adj Close', 'Next Day Adj Close', 'Moving Average', 'Momentum'], axis=1 , inplace=True)
        return features
    
    """Returns The Ticker label"""
    def get_label(self):
        df = self._get_clean_df()
        label = df['Next Day Adj Close']
        return label

    """Get a dataframe with next day adjusted close price"""
    def get_next_day_df(self):
        df = self._get_clean_df()
        return df[['Next Day Adj Close']]
    
    """Add next daty adjusted close price"""
    def _add_next_day_price(self, df=None):
        return (df.shift(-1)).rename(columns={'Adj Close': 'Next Day Adj Close'})
    
    """Print the ticker data as a DataFrame"""
    def print_df(self):
        print self.df  

    """Return a cleaned DataFrame by replacing or remiving missing data"""
    def get_df(self):
          return self._get_clean_df()
    
    """Calculate the ticker momentum"""
    def _calculate_momentum(self, df=None):
        n=self.momentum_window
        
        df=df/df.shift(n)
        return (df[n:] - 1).rename(columns={'Adj Close': 'Momentum'})

    """Calculate the ticker volatility"""
    def _calculate_volatility(self, df=None):
        n=self.volatility_window
        return ((df.rolling(window=n,center=False).std())[n:] - 1 ).rename(columns={'Adj Close': 'Volatility'})
   
    """Calculate the ticker Bollinger Bands"""
    def _calculate_bb(self, df=None):
        n=self.bollinger_bands_window
        
        std=(df.rolling(window=n,center=False).std())[n:]
        ma=(df.rolling(window=n,center=False).mean())[n:]
        upper_bound=ma+2*std
        lower_bound=ma-2*std
        return upper_bound.rename(columns={'Adj Close': 'BB Upper Bound'}), lower_bound.rename(columns={'Adj Close': 'BB Lower Bound'})

    """Calculate the ticker Simple Moving Average"""
    def _calculate_moving_average(self, df=None):
        n=self.moving_average_window
        
        return ((df/df.rolling(window=n,center=False).mean())[n:] - 1).rename(columns={'Adj Close': 'Moving Average'})

    """Calculate the ticker Daily Returns"""
    def _calculate_daily_returns(self, df=None):
        daily_returns=df.copy()
        daily_returns=(daily_returns/daily_returns.shift(1))-1
        return daily_returns[1:].rename(columns={'Adj Close': 'Daily Returns'})