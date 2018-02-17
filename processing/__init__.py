"""Data Processing"""

import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader as pdr
import pandas_datareader.data as web
import numpy as np

from IPython.display import display # Allows the use of display() for DataFrames

# Pretty display
%matplotlib inline

style.use('ggplot')

"""Retrieve S&P data from January 2000  to December 2016"""
def funcion run():
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2016, 12, 31)
    df = pdr.get_data_yahoo('TSLA', start, end)
    df.to_csv('tesla.csv')

    print(df.head())



if __name__ == "__main__":
    run()