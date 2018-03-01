import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math

from sklearn import svm, neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from statistics.Portfolio import Portfolio
from prediction.Ticker import TickerAnalysed, Ticker
from statistics.Sim import Sim
from common.start import get_prices, get_all_tickers, visualize_correlation, plot_data, get_tickers_for_symbols
from common.start import store_pickle, get_pickle, store_ticker_analysis, CLASSIFIER_PICKLE_DIR

def get_classifiers():
    classifiers_needing_scaling = ['SVM Regression Linear', 'SVM Regression Poly', 'SVM Regression RBF']
    classifier_names = ['Nearest Neighbor Regressor', 'Random Forest Regressor', 'SVM Regression Linear',
        'SVM Regression Poly', 'SVM Regression RBF', 'Linear Regression']
    classifiers = [neighbors.KNeighborsRegressor(), RandomForestRegressor(), svm.SVR(kernel='linear', C=1e3), 
        svm.SVR(kernel= 'poly', C=1e3, degree=2), svm.SVR(kernel='rbf', C=1e3, gamma=0.1), LinearRegression()]
    return dict(zip(classifier_names, classifiers)), classifiers_needing_scaling

def run_prediction(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)

    return confidence, predictions

def cross_validate(ticker, clf_name, clf):
    X = ticker.get_features()
    y = ticker.get_label()

    tscv = TimeSeriesSplit(n_splits = 3)

    confidenceList = []
    rmseList = []
    classifiers, need_scaling = get_classifiers()
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if clf_name in need_scaling:
            print clf_name + ' needs scaling'
            X_train_scaled = StandardScaler().fit_transform(X_train)
            X_test_scaled = StandardScaler().fit_transform(X_test)
            confidence, predictions = run_prediction(clf, X_train_scaled, X_test_scaled, y_train, y_test)
        else:
            print clf_name + ' does not need scaling'
            confidence, predictions = run_prediction(clf, X_train, X_test, y_train, y_test)
        rmseList.append(math.sqrt(((y_test - predictions) ** 2).sum()/len(y_test)))
        confidenceList.append(confidence)

    confidence_mean = np.array(confidenceList).mean()
    rmse_mean = np.array(rmseList).mean()
    print '----------------------------------------------------------------'
    print 'Results for classifier: ' + clf_name
    print '{} Confidence: {:,.3f}'.format(ticker.symbol, confidence_mean)
    print "{} Root Mean Squared Error of Predictions: {:,.3f}".format(ticker.symbol, rmse_mean)
    print '----------------------------------------------------------------'

def get_model_pickle_path(symbol, clf_name):
    _clf_name = clf_name.replace(' ', '_').lower()
    return symbol+'_'+_clf_name+'_model.sav'

def generate_model(ticker, clf_name, clf):
    X_train = ticker.get_features()
    y_train = ticker.get_label()
    classifiers, need_scaling = get_classifiers()
    if clf_name in need_scaling:
        X_train_scaled = StandardScaler().fit_transform(X_train)
        clf.fit(X_train_scaled, y_train)
    else:
        clf.fit(X_train, y_train)
   
    # pickle the classifier for use at a later time
    store_pickle(clf, get_model_pickle_path(ticker.symbol, clf_name), CLASSIFIER_PICKLE_DIR)

def run_classifier_for_symbol(ticker, clf_name):
    X_predict = ticker.get_features()
    model = get_pickle(get_model_pickle_path(ticker.symbol, clf_name), CLASSIFIER_PICKLE_DIR)

    return model.predict(X_predict)

def predict(ticker, clf_name):
    predict_df = ticker.get_adj_close_df().copy()
    predict_df['Predictions'] = run_classifier_for_symbol(ticker, clf_name)
    _clf_name = clf_name.replace(' ', '_').lower()
    store_ticker_analysis(predict_df, ticker.symbol+'_'+_clf_name+'_prediction')
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
    classifiers, need_scaling = get_classifiers()
    for clf_name, clf in classifiers.iteritems():
        prediction_df = predict(ticker, clf_name)
        print symbol + 'Predictions using : ' + clf_name
        print prediction_df[window-2:]

def predict_for_symbols(symbols, start_date, end_date):
    prices_df, prices_df_with_spy = get_prices(symbols, start_date, end_date)
    classifiers, need_scaling = get_classifiers()
    for symbol in symbols:
        ticker = TickerAnalysed(symbol=symbol, data_df=prices_df[[symbol]])
        
        for clf_name, clf in classifiers.iteritems():
            # analyze_features(ticker)
            cross_validate(ticker, clf_name, clf)
            generate_model(ticker, clf_name, clf)
            prediction_df = predict(ticker, clf_name)
            # plot_data(prediction_df.tail(10), title="Prediction vs actual")

def run(): 
    train_start_date ='2017-01-03'
    train_end_date = '2017-11-03'
    buy_date = '2017-11-06'
    sell_date = '2017-11-10'
    investment = 10000 # $10,000.00 as starting investment
    hold_spy(investment, buy_date, sell_date)
    hold_optimized_portfolio(investment, buy_date, sell_date)

    # train and test the model
    symbols = ['PGR', 'CCI', 'STZ', 'WYNN', 'TPR', 'DPS']
    predict_for_symbols(symbols, train_start_date, train_end_date)

    use_predictions_optimized_portfolio(investment, buy_date, sell_date)
    
   
if __name__ == "__main__":
    run()