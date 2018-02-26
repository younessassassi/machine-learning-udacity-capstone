import numpy as np
import pandas as pd

from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from prediction.Ticker import Ticker
import matplotlib.pyplot

from common.start import get_ticker_data, get_all_tickers
from common.start import store_pickle, get_pickle

# def siraj():
#     svr_lin = svm.SVR(kernal='linear', C=1e3)
#     svr_poly = svm.SVR(kernel= 'poly', C=1e3, degree=2)
#     svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)

#     svr_lin.fit(dates, prices)
#     svr_poly.fit(dates, prices)
#     svr_rbf.fit(dates, prices)

#     plt.scatter(dates, prices, color='black', label='Data')
#     plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
#     plt.plot(dates, svr_lin.predict(dates), color='green', label='linear model')
#     plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')
#     plt.xlabel('Date')
#     plt.ylabel('Price')
#     plt.title('Support Vector Regression')
#     plt.legend()
#     plt.show()

#     return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

# def sentdexpredict(ticker):
#     X, y, df = extract_featuresets(ticker)

#     X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size:0.25)

#     # clf = neighbors.KNeighborsClassifier()
#     clf = VotingClassifier([('lsvc', svm.LinearSVC()),
#                             ('knn', neighbors.KNeighborsClassifier()),
#                             ('rfor', RandomForestClassifier())])
#     clf.fit(X_train, y_train)
#     confidence = clf.score(X_test, y_test)
#     # pickle the classifier for use at a later time
#     classifier_path = CLASSIFIER_PICKLE_DIR + 'knn'
#     store_pickle(classifier_path, clf)
#     print 'Accuracy: ' + confidence
#     predictions = clf.predict(X_test)
#     print 'Predicted spread: ', Counter(predictions)

#     return confidence

# def score(clf, X, y, cv=2):
#     scores = cross_val_score(clf, X, y, cv=2)

#     print 'scores: ', scores

#     print 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2)

def run_prediction(X_train, X_test, y_train, y_test):
    # svr_lin = svm.SVR(kernal='linear', C=1e3)
    # svr_poly = svm.SVR(kernel= 'poly', C=1e3, degree=2)
    # svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    # clf = VotingClassifier([('lsvc', svm.LinearSVC()),
    #                         ('knn', neighbors.KNeighborsClassifier()),
    #                         ('rfor', RandomForestClassifier())])
    clf =  LinearRegression()
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    # pickle the classifier for use at a later time
    # classifier_path = CLASSIFIER_PICKLE_DIR + 'knn'
    # store_pickle(classifier_path, clf)
    print 'Accuracy: ', confidence
    predictions = clf.predict(X_test)
    print 'Predictions: ', predictions

    return confidence

def predict_for_symbol(symbol, stocks, start_date, end_date):
    ticker = Ticker(symbol=symbol, stocks=stocks, 
                 start_date=start_date, end_date=end_date)
    X = ticker.get_features()
    y = ticker.get_label()

    tscv = TimeSeriesSplit(n_splits = 3)

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        confidence = run_prediction(X_train, X_test, y_train, y_test)

    # score(clf, X, y, cv)

    

def predict():
    start_date = '2011-01-01'
    end_date = '2011-01-30'
    tickers = ['T']
    stocks = get_ticker_data(None, start_date, end_date)
    
    for symbol in tickers:
        predict_for_symbol(symbol, stocks, start_date, end_date)
    
   
    

def run():
    predict()

if __name__ == "__main__":
    run()