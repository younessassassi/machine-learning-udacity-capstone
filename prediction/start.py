import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn import svm, neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from statistics.Portfolio import Portfolio
from prediction.Ticker import TickerAnalysed, Ticker
from statistics.Sim import Sim
from statistics.start import get_allocations_used, optimize_portfolio
from common.start import get_prices, get_all_tickers, visualize_correlation, plot_data, get_tickers_for_symbols
from common.start import store_pickle, get_pickle, store_ticker_analysis, store_classifer_analysis
from common.start import get_classifier_analysis, CLASSIFIER_PICKLE_DIR


def get_classifiers(): 
    classifiers_needing_scaling = ['SVM Regression Linear', 'SVM Regression Poly', 'SVM Regression RBF']
    classifier_names = ['Nearest Neighbor Regressor', 'Random Forest Regressor', 'SVM Regression Linear',
        'SVM Regression Poly', 'SVM Regression RBF', 'Linear Regression']
    classifiers = [neighbors.KNeighborsRegressor(), RandomForestRegressor(max_depth=2, random_state=0), svm.SVR(kernel='linear', C=1e3), 
        svm.SVR(kernel= 'poly', C=1e3, degree=2), svm.SVR(kernel='rbf', C=1e3, gamma=0.1), LinearRegression()]
    return dict(zip(classifier_names, classifiers)), classifiers_needing_scaling

def run_prediction(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)

    return confidence, predictions, clf

def cross_validate(ticker, clf_name, clf):
    X = ticker.get_features()
    y = ticker.get_label()

    tscv = TimeSeriesSplit(n_splits = 3)

    confidence_list = []
    rmse_list_out = []
    rmse_list_in = []
    classifiers, need_scaling = get_classifiers()
    actual_values = []
    predicted_values = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        actual_values.append(X_test)

        if clf_name in need_scaling:
            X_train_scaled = StandardScaler().fit_transform(X_train)
            X_test_scaled = StandardScaler().fit_transform(X_test)
            confidence, predictions, clf = run_prediction(clf, X_train_scaled, X_test_scaled, y_train, y_test)
        else:
            confidence, predictions, clf = run_prediction(clf, X_train, X_test, y_train, y_test)
        confidence_list.append(confidence)
        rmse_list_out.append(math.sqrt(((y_test - clf.predict(X_test)) ** 2).sum()/len(y_test)))
        rmse_list_in.append(math.sqrt(((y_train - clf.predict(X_train)) ** 2).sum()/len(X_test)))
        predicted_values.append(predictions)
        
    confidence_mean = np.array(confidence_list).mean()
    rmse_mean_in = np.array(rmse_list_in).mean()
    rmse_mean_out = np.array(rmse_list_out).mean()
    print '---------------------------------------------------------------'
    print 'Results for classifier: ' + clf_name    
    print '{} Confidence: {:,.3f}'.format(ticker.symbol, confidence_mean)
    print "{} In sample Root Mean Squared Error: {:,.3f}".format(ticker.symbol, rmse_mean_in)
    print "{} Out of sample Root Mean Squared Error: {:,.3f}".format(ticker.symbol, rmse_mean_out)
    print '----------------------------------------------------------------'
    return [ticker.symbol, clf_name, confidence_mean, rmse_mean_in, rmse_mean_out]


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
    predict_df = ticker.df[['Next Day Adj Close']].copy() 
    predict_df['Predictions'] = run_classifier_for_symbol(ticker, clf_name)
    _clf_name = clf_name.replace(' ', '_').lower()
    # remove last row as we dont have the next day to compare against
    predict_df = predict_df[:-1]
    store_ticker_analysis(predict_df, ticker.symbol+'_'+_clf_name+'_prediction')
    return predict_df

def analyze_features(ticker):
    ticker_df = ticker.get_df()
    store_ticker_analysis(ticker_df, ticker.symbol)
    visualize_correlation(ticker_df, 'Variable Correlation')

def get_portfolio(symbols, weights, start_date, end_date, investment, no_spy=True):
    prices_df, prices_df_with_spy = get_prices(symbols, start_date, end_date, no_spy)
    tickers = get_tickers_for_symbols(symbols, start_date, end_date, no_spy)
    portfolio = Portfolio(tickers, weights, start_date, end_date, investment)

    return portfolio

def simulate_trade(symbols, weights, buy_date, sell_date, investment, no_spy=True):
    portfolio = get_portfolio(symbols, weights, buy_date, sell_date, investment, no_spy)
    simulation = Sim(portfolio, buy_date, sell_date)
    # print '---------------------------'
    # print 'Simulation for'
    # print '---------------------------'
    # portfolio.describe()
    # print '---------------------------'
   
    return simulation

def hold_spy(investment, buy_date, sell_date):
    symbols = ['SPY']
    weights = [1.0]
    simulation = simulate_trade(symbols, weights, buy_date, sell_date, investment, no_spy=False)
    print 'The cash value for holding the S&P 500 from {} to {} is {}'.format(buy_date, sell_date, simulation.cash_out())

def hold_optimized_portfolio(investment, buy_date, sell_date):
    symbols = ['PGR', 'CCI', 'STZ', 'WYNN', 'TPR', 'DPS']
    weights = [0.40, 0.21, 0.19, 0.12, 0.05, 0.03]
    simulation = simulate_trade(symbols, weights, buy_date, sell_date, investment)
    print 'The cash value for holding the original optimized portfolio from {} to {} is {}'.format(buy_date, sell_date, simulation.cash_out())

def get_optimized_portfolio(symbols, start_date, end_date, investment):
    # get new optimized portfolio and get its value for the date
    tickers = get_tickers_for_symbols(symbols, start_date, end_date)
    weights = []
    for symbol in symbols:
         weights.append(1/float(len(symbols)))
    portfolio = Portfolio(tickers, weights, start_date, end_date, investment)
    optimized_portfolio = optimize_portfolio(portfolio)
    # optimized_portfolio.describe()
    return optimized_portfolio

def use_predictions_optimized_portfolio(investment, buy_date, sell_date, train_start, train_end):
    # check for each day of the trade if symbol adds value, if so buy otherwise hold
    symbols = ['PGR', 'CCI', 'STZ', 'WYNN', 'TPR', 'DPS']
    cash = investment
    start_date = pd.to_datetime(buy_date)

    end_date = pd.to_datetime(sell_date)
    one_before_end_date = end_date - timedelta(days=1)
    date_range = pd.date_range(start_date, end_date)
    simulation = None
    for date in date_range:
        predicted_symbols = []
        if date < one_before_end_date:
            date_plus_one = date + timedelta(days=1)
            for symbol in symbols:
                # print 'start date: ', date
                # print 'end date: ' , date_plus_one
                prediction_df, original_prices_df = predict_for_symbol([symbol], date, date_plus_one)
                # print 'Orginal for ' + symbol, original_prices_df
                # if (prediction_df[date:'Predictions'])
                # print 'first date: ', prediction_df.loc[date]['Predictions']

                if prediction_df.loc[date]['Predictions'] > original_prices_df.loc[date]['Adj Close']:
                    # if symbol is predicted to rise then add it to new portfolio
                    predicted_symbols.append(symbol)
            # get new optimized portofolio
            optim_port = get_optimized_portfolio(predicted_symbols, train_start, train_end, cash)
            # get value of new optimaized portfolio
            simulation = simulate_trade(optim_port.symbols, optim_port.weights, date, date_plus_one, cash)
            # update investment value
            cash = simulation.cash_out()
      
    print 'The cash value for trading based on Regression Model from {} to {} is {}'.format(buy_date, sell_date, simulation.cash_out())

def predict_for_symbol(symbols, start_date, end_date, clf_name="Linear Regression"):
    window = 5
    symbol = symbols[0]
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    start_date = start_date - timedelta(days=window)
    end_date = end_date + timedelta(days=1)
    
    prices_df, prices_df_with_spy = get_prices(symbols, start_date, end_date)
    ticker = TickerAnalysed(symbol=symbol, data_df=prices_df[[symbol]])
    # analyze_features(ticker)
            
    # classifiers, need_scaling = get_classifiers()
    prediction_df = predict(ticker, clf_name)
    # for clf_name, clf in classifiers.iteritems():
    prediction_df = predict(ticker, clf_name)
    # print symbol + ' Predictions using : ' + clf_name
    # print prediction_df[window-2:]
    # # plot_data(prediction_df, title=symbol + " Prediction vs actual")
    return prediction_df[window-2:], ticker.get_adj_close_df()[window-2:]
        
def predict_for_symbols(symbols, start_date, end_date):
    prices_df, prices_df_with_spy = get_prices(symbols, start_date, end_date)
    classifiers, need_scaling = get_classifiers()
    classifier_results = []
    for symbol in symbols:
        ticker = TickerAnalysed(symbol=symbol, data_df=prices_df[[symbol]])
        
        for clf_name, clf in classifiers.iteritems():
            classifier_results.append(cross_validate(ticker, clf_name, clf))
            generate_model(ticker, clf_name, clf)
    return pd.DataFrame(classifier_results, columns=['Symbol', 'Classifier', 'Confidence',
        'RMSE In Sample', 'RMSE Out of Sample'])
   
        
        # prediction_df = predict(ticker, clf_name)
        # plot_data(prediction_df.tail(10), title="Prediction vs actual")

def visualize_classifier_results():
    df = get_classifier_analysis()
    df.drop(df.columns[0], axis=1, inplace=True)
    symbols = df['Symbol']
    df = df[df['Symbol'] == symbols[0]]
    df = df[df['Classifier'] != 'SVM Regression Poly']
    df.drop('Symbol', axis=1, inplace=True)
    df.set_index('Classifier', inplace=True)
    df.sort_values(by=['RMSE Out of Sample'], inplace=True)
    print df
    df_rmse = df[['RMSE Out of Sample']]
    print df_rmse
    
    
    ax = df_rmse.plot(kind='bar', title ="Classifier Comparison", figsize=(15, 10), legend=True, fontsize=10)
    ax.set_xlabel("Classifier", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    plt.xticks(rotation='horizontal')
    plt.show()

def run(): 
    train_start_date ='2017-01-03'
    train_end_date = '2017-11-03'
    buy_date = '2017-11-06'
    sell_date = '2017-11-10'
    investment = 10000 # $10,000.00 as starting investment
    hold_spy(investment, buy_date, sell_date)
    hold_optimized_portfolio(investment, buy_date, sell_date)
    use_predictions_optimized_portfolio(investment, buy_date, sell_date, train_start_date, train_end_date)
    # train and test the model
    # symbols = ['PGR', 'CCI', 'STZ', 'WYNN', 'TPR', 'DPS']
    # df = predict_for_symbols(symbols, train_start_date, train_end_date)
    # store_classifer_analysis(df)

    # use_predictions_optimized_portfolio(investment, buy_date, sell_date)
    # visualize_classifier_results()    
   
if __name__ == "__main__":
    run()