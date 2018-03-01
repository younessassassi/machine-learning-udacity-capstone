import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math

from sklearn import svm, neighbors
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from statistics.Portfolio import Portfolio
from prediction.Ticker import TickerAnalysed, Ticker
from statistics.Sim import Sim
from common.start import get_prices, get_all_tickers, visualize_correlation, plot_data, get_tickers_for_symbols
from common.start import store_pickle, get_pickle, store_ticker_analysis, CLASSIFIER_PICKLE_DIR

def run_prediction(X_train, X_test, y_train, y_test):
    # svr_lin = svm.SVR(kernel='linear', C=1e3)
    # svr_poly = svm.SVR(kernel= 'poly', C=1e3, degree=2)
    # svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    # clf = svr_lin
    # clf = svr_poly
    # clf = svr_rbf
    clf =  LinearRegression()
    # clf = neighbors.KNeighborsRegressor()
    # clf = RandomForestRegressor()
    # clf = neighbors.KNeighborsRegressor()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)

    return confidence, predictions

def cross_validate(ticker):
    X = ticker.get_features()
    y = ticker.get_label()

    tscv = TimeSeriesSplit(n_splits = 3)

    confidenceList = []
    rmseList = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        confidence, predictions = run_prediction(X_train, X_test, y_train, y_test)
        rmseList.append(math.sqrt(((y_test - predictions) ** 2).sum()/len(y_test)))
        confidenceList.append(confidence)

    confidence_mean = np.array(confidenceList).mean()
    rmse_mean = np.array(rmseList).mean()
    print 'Confidence: ', confidence_mean
    print "Root Mean Squared Error of Predictions: ", rmse_mean

def generate_model(ticker):
    X_train = ticker.get_features()
    y_train = ticker.get_label()
    # clf = neighbors.KNeighborsRegressor()
    clf = LinearRegression()
    # clf = RandomForestRegressor()
    clf.fit(X_train, y_train)
    # pickle the classifier for use at a later time
    classifier_path = CLASSIFIER_PICKLE_DIR
    store_pickle(clf, ticker.symbol+'_model.sav', classifier_path)

def run_classifier_for_symbol(ticker):
    X_predict = ticker.get_features()
    classifier_path = CLASSIFIER_PICKLE_DIR
    model = get_pickle(ticker.symbol+'_model.sav', classifier_path)

    return model.predict(X_predict)

def predict(ticker):
    predict_df = ticker.get_adj_close_df().copy()
    predict_df['Predictions'] = run_classifier_for_symbol(ticker)
    store_ticker_analysis(predict_df, 'prediction_'+ticker.symbol)
    return predict_df

def analyze_features(ticker):
    ticker_df = ticker.get_df()
    store_ticker_analysis(ticker_df, ticker.symbol)
    visualize_correlation(ticker_df)

def get_portfolio(symbols, weights, start_date, end_date, investment, no_spy=True):
    prices_df, prices_df_with_spy = get_prices(symbols, start_date, end_date, no_spy)
    tickers = get_tickers_for_symbols(symbols, start_date, end_date, no_spy)
    portfolio = Portfolio(tickers, weights, start_date, end_date, investment)
    return portfolio

def simulate_trade(symbols, weights, buy_date, sell_date, investment, no_spy=True):
    portfolio = get_portfolio(symbols, weights, buy_date, sell_date, investment, no_spy)
    simulation = Sim(portfolio, buy_date, sell_date)
    print '---------------------------'
    print 'Simulation for'
    print '---------------------------'
    portfolio.describe()
    print '---------------------------'
   
    return simulation.cash_out(buy_date, sell_date)

def hold_spy(investment, buy_date, sell_date):
    symbols = ['SPY']
    weights = [1.0]
    return simulate_trade(symbols, weights, buy_date, sell_date, investment, no_spy=False)

def hold_optimized_portfolio(investment, buy_date, sell_date):
    symbols = ['PGR', 'CCI', 'STZ', 'WYNN', 'TPR', 'DPS']
    weights = [0.40, 0.21, 0.19, 0.12, 0.05, 0.03]
    return simulate_trade(symbols, weights, buy_date, sell_date, investment)


def use_predictions_optimized_portfolio(investment, buy_date, sell_date):
    symbols = ['PGR', 'CCI', 'STZ', 'WYNN', 'TPR', 'DPS']
    weights = [0.40, 0.21, 0.19, 0.12, 0.05, 0.03]
    # predict_for_symbols(symbols, buy_date, sell_date)
    for symbol in symbols:
        predict_for_symbol([symbol], buy_date, sell_date)
        
    # return simulate_trade(symbols, weights, buy_date, sell_date, investment)

def predict_for_symbol(symbols, start_date, end_date):
    window = 5
    symbol = symbols[0]
    start_date = pd.to_datetime(start_date)
    start_date = start_date - timedelta(days=window)

    prices_df, prices_df_with_spy = get_prices(symbols, start_date, end_date)
    ticker = TickerAnalysed(symbol=symbol, data_df=prices_df[[symbol]])
    prediction_df = predict(ticker)
    print 'Predictions: '
    print prediction_df[window-2:]

def predict_for_symbols(symbols, start_date, end_date):
    prices_df, prices_df_with_spy = get_prices(symbols, start_date, end_date)
    for symbol in symbols:
        ticker = TickerAnalysed(symbol=symbol, data_df=prices_df[[symbol]])
        # analyze_features(ticker)
        cross_validate(ticker)
        generate_model(ticker)
        prediction_df = predict(ticker)
        # plot_data(prediction_df.tail(10), title="Prediction vs actual")

def run(): 
    train_start_date ='2017-01-03'
    train_end_date = '2017-11-03'
    buy_date = '2017-11-06'
    sell_date = '2017-11-10'
    investment = 100000 # $100,000.00 as starting investment
    hold_spy(investment, buy_date, sell_date)
    hold_optimized_portfolio(investment, buy_date, sell_date)

    # train and test your model
    symbols = ['PGR', 'CCI', 'STZ', 'WYNN', 'TPR', 'DPS']
    predict_for_symbols(symbols, train_start_date, train_end_date)

    use_predictions_optimized_portfolio(investment, buy_date, sell_date)
    
   
if __name__ == "__main__":
    run()