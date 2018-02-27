import numpy as np
import pandas as pd
import math

from sklearn import svm, neighbors
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from prediction.Ticker import Ticker
import matplotlib.pyplot

from common.start import get_ticker_data, get_all_tickers, visualize_correlation
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
    # pickle the classifier for use at a later time
    # classifier_path = CLASSIFIER_PICKLE_DIR + 'knn'
    # store_pickle(classifier_path, clf)
    # print 'Accuracy: ', confidence
    predictions = clf.predict(X_test)
    # print 'Predictions: ', predictions

    return confidence, predictions

def cross_validate_for_symbol(symbol, stocks, start_date, end_date):
    ticker = Ticker(symbol=symbol, stocks=stocks, 
                 start_date=start_date, end_date=end_date)
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

def create_classifier_for_symbol(symbol, stocks, start_date, end_date):
    ticker = Ticker(symbol=symbol, stocks=stocks, 
                 start_date=start_date, end_date=end_date)
    X_train = ticker.get_features()
    y_train = ticker.get_label()
    # clf = neighbors.KNeighborsRegressor()
    clf = LinearRegression()
    # clf = RandomForestRegressor()
    clf.fit(X_train, y_train)
    # pickle the classifier for use at a later time
    classifier_path = CLASSIFIER_PICKLE_DIR
    store_pickle(clf, symbol+'_model.sav', classifier_path)

def plot(actual_prices, predictions):
    print 'Actual prices: ', actual_prices
    # data = {'Actual Prices': actual_prices, 'Predictions': predictions}
    # # print data
    # df = pd.DataFrame(data = data)
    # ax = df.plot(title="Prediction Results", fontsize=12)
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Price")
    # plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    # plt.show()

def run_classifier_for_symbol(symbol, stocks, start_date, end_date):
    ticker = Ticker(symbol=symbol, stocks=stocks, 
                 start_date=start_date, end_date=end_date)
    X_predict = ticker.get_features()
    classifier_path = CLASSIFIER_PICKLE_DIR
    model = get_pickle(symbol+'_model.sav', classifier_path)

    return model.predict(X_predict)

def run_model_validation(tickers, start_date, end_date):
    stocks = get_ticker_data(None, start_date, end_date)
    for symbol in tickers:
        cross_validate_for_symbol(symbol, stocks, start_date, end_date)
       
def generate_model(tickers, start_date, end_date):
    stocks = get_ticker_data(None, start_date, end_date)
    for symbol in tickers:
        create_classifier_for_symbol(symbol, stocks, start_date, end_date)


def predict(tickers, start_date, end_date):
    stocks = get_ticker_data(None, start_date, end_date)
    for symbol in tickers:
        predictions = run_classifier_for_symbol(symbol, stocks, start_date, end_date)
        actual_prices = get_ticker_data([symbol], start_date, end_date)
        print 'predictions for {} from {} to {}'.format(symbol, start_date, end_date)
        print  predictions
        print 'actual prices for {} from {} to {}'.format(symbol, start_date, end_date)
        print actual_prices

def analyze_features(tickers, start_date, end_date):
    stocks = get_ticker_data(None, start_date, end_date)
    dates = pd.date_range(start_date, end_date)

    for symbol in tickers:
        ticker = Ticker(symbol=symbol, stocks=stocks, 
                 start_date=start_date, end_date=end_date)
        ticker_df = ticker.get_df()
        store_ticker_analysis(ticker_df, symbol)
        visualize_correlation(ticker_df)
    

def run():
    start_date = '2014-01-01'
    end_date = '2018-02-15'
    tickers = ['AAPL', 'T']
    # analyze_features(tickers, start_date, end_date)
    run_model_validation(tickers, start_date, end_date)
    # generate_model(tickers, start_date, end_date)
    # start_date = '2017-12-24'
    # end_date = '2018-01-03'
    # analyze_features(tickers, start_date, end_date)
    # predict(tickers,start_date, end_date)
    

if __name__ == "__main__":
    run()