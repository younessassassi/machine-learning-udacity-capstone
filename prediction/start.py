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
from common.start import store_optimized_portfolio, get_optimized_portfolio

"""Get portfolio return"""
def get_portfolio_return(begining_value, ending_value):
    return (ending_value - begining_value)/begining_value

"""Get a dictionary of various classifers with their corresponding long name"""
# def get_classifiers(): 
#     classifiers_needing_scaling = []
#     classifier_names = ['Linear Regression']
#     classifiers = [LinearRegression()]
#     return dict(zip(classifier_names, classifiers)), classifiers_needing_scaling


def get_classifiers(): 
    classifiers_needing_scaling = ['SVM Regression Linear', 'SVM Regression Poly', 'SVM Regression RBF']
    classifier_names = ['Nearest Neighbor Regressor', 'Random Forest Regressor', 'SVM Regression Linear',
        'SVM Regression Poly', 'SVM Regression RBF', 'Linear Regression']
    classifiers = [neighbors.KNeighborsRegressor(), RandomForestRegressor(max_depth=2, random_state=0), svm.SVR(kernel='linear', C=1e3), 
        svm.SVR(kernel= 'poly', C=1e3, degree=2), svm.SVR(kernel='rbf', C=1e3, gamma=0.1), LinearRegression()]
    return dict(zip(classifier_names, classifiers)), classifiers_needing_scaling

"""Train the model and return its confidence score and predictions"""
def run_prediction(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)

    return confidence, predictions, clf

"""Run a cross validation for a classifier"""
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
        
    # print 'predictions: ',  predictions
    # print 'y test: ', y_test
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

"""Get the classifier path name"""
def get_model_pickle_path(symbol, clf_name):
    _clf_name = clf_name.replace(' ', '_').lower()
    return symbol+'_'+_clf_name+'_model.sav'

"""Generate a model per classifier and ticker symbol then store model for later use"""
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

"""Use a stored model to predict future price for a ticker"""
def run_classifier_for_symbol(ticker, clf_name):
    X_predict = ticker.get_features()
    model = get_pickle(get_model_pickle_path(ticker.symbol, clf_name), CLASSIFIER_PICKLE_DIR)

    return model.predict(X_predict)

"""store and return predictions using a model and ticker"""
def predict(ticker, clf_name):
    predict_df = ticker.df[['Next Day Adj Close']].copy() 
    predict_df['Predictions'] = run_classifier_for_symbol(ticker, clf_name)
    _clf_name = clf_name.replace(' ', '_').lower()
    # remove last row as we dont have the next day to compare against
    predict_df = predict_df[:-1]
    store_ticker_analysis(predict_df, ticker.symbol+'_'+_clf_name+'_prediction')
    return predict_df

"""Visulize the correlation between different ticker features"""
def analyze_features(ticker):
    ticker_df = ticker.get_df()
    store_ticker_analysis(ticker_df, ticker.symbol)
    visualize_correlation(ticker_df, 'Variable Correlation')

"""Get a portfolio based on some input data"""
def get_portfolio(symbols, weights, start_date, end_date, investment, no_spy=True):
    prices_df, prices_df_with_spy = get_prices(symbols, start_date, end_date, no_spy)
    tickers = get_tickers_for_symbols(symbols, start_date, end_date, no_spy)
    portfolio = Portfolio(tickers, weights, start_date, end_date, investment)

    return portfolio

"""Retrurn a simulation object for a set of symbols, weights and starting investment"""
def simulate_trade(symbols, weights, buy_date, sell_date, investment, no_spy=True):
    portfolio = get_portfolio(symbols, weights, buy_date, sell_date, investment, no_spy)
    simulation = Sim(portfolio, buy_date, sell_date)
    # print '---------------------------'
    # print 'Simulation for'
    # print '---------------------------'
    # portfolio.describe()
    # print '---------------------------'
   
    return simulation

"""Simulation holding the S&P 500 ticker during a date range"""
def hold_spy(investment, buy_date, sell_date):
    symbols = ['SPY']
    weights = [1.0]
    simulation = simulate_trade(symbols, weights, buy_date, sell_date, investment, no_spy=False)
    print 'The cash value for holding the S&P 500 from {} to {} is ${:,.2f}'.format(buy_date, sell_date, simulation.cash_out())
    print 'The return value for holding the S&P 500 from {} to {} is {:,.2f}%'.format(buy_date, sell_date, (get_portfolio_return(investment, simulation.cash_out())) * 100)
    return simulation.get_daily_returns(), simulation.get_value()

"""Simulation holding the S&P 500 ticker during a date range"""
def hold_optimized_portfolio(investment, buy_date, sell_date, symbols, weights):
    simulation = simulate_trade(symbols, weights, buy_date, sell_date, investment)
    print 'The cash value for holding the original optimized portfolio from {} to {} is ${:,.2f}'.format(buy_date, sell_date, simulation.cash_out())
    print 'The return value for holding the original optimized portfolio from {} to {} is {:,.2f}%'.format(buy_date, sell_date, (get_portfolio_return(investment, simulation.cash_out())) * 100)
    return simulation.get_daily_returns(), simulation.get_value()

"""Generate an optimized portfolio using historical data and symbols"""
def generate_optimized_portfolio(symbols, start_date, end_date, investment, rerun=False):
    # get new optimized portfolio and get its value for the date
    tickers = get_tickers_for_symbols(symbols, start_date, end_date)
    weights = []
    for symbol in symbols:
        weights.append(1/float(len(symbols)))
    portfolio = Portfolio(tickers, weights, start_date, end_date, investment)
    optimized_portfolio = optimize_portfolio(portfolio)
    # only keep symbols with 2% or higher weight and only run it once to make sure we dont create an inifinite loop
    if not rerun:
        # create dict of weights and symbols
        symb_weight_dict = dict(zip(optimized_portfolio.symbols, optimized_portfolio.weights))
        symbols_to_keep = []
        for item in symb_weight_dict:
            if symb_weight_dict[item] > 0.02:
                symbols_to_keep.append(item)
            else:
                pass
        if len(symbols_to_keep) < len(symbols):
            print 'running optimizer one more time'
            optimized_portfolio = generate_optimized_portfolio(symbols_to_keep, start_date, end_date, investment, True)

    return optimized_portfolio

"""Simulate the trading of an optimized portfolio"""
def use_predictions_optimized_portfolio(investment, buy_date, sell_date, train_start, train_end, symbols):
    # check for each day of the trade if symbol adds value, if so buy 
    cash = investment
    start_date = pd.to_datetime(buy_date)

    end_date = pd.to_datetime(sell_date)
    one_before_end_date = end_date - timedelta(days=1)
    date_range = pd.date_range(start_date, end_date)
    simulation = None
    daily_returns = np.array([0.0])
    value = np.array([cash])
    for date in date_range:
        predicted_symbols = []
        if date < end_date:
            date_plus_one = date + timedelta(days=1)
            for symbol in symbols:
                # print 'start date: ', date
                # print 'end date: ' , date_plus_one
                prediction_df, original_prices_df = predict_for_symbol([symbol], date, date_plus_one)  
                # if (prediction_df[date:'Predictions'])
                # print 'first date: ', prediction_df.loc[date]['Predictions']

                if prediction_df.loc[date]['Predictions'] > original_prices_df.loc[date]['Adj Close']:
                    # if symbol is predicted to rise then add it to new portfolio
                    predicted_symbols.append(symbol)
            # get new optimized portofolio
            optim_port = generate_optimized_portfolio(predicted_symbols, train_start, train_end, cash)
            # get value of new optimaized portfolio
            simulation = simulate_trade(optim_port.symbols, optim_port.weights, date, date_plus_one, cash)
            # update investment value
            cash = simulation.cash_out()
            daily_returns = np.concatenate((daily_returns, simulation.get_daily_returns()[1:]), axis=0)
            value = np.concatenate((value, simulation.get_value()[1:]), axis=0)
    print 'The cash value for trading based on Regression Model from {} to {} is ${:,.2f}'.format(buy_date, sell_date, simulation.cash_out())
    print 'The return value for trading based on Regression Model from {} to {} is {:,.2f}%'.format(buy_date, sell_date, (get_portfolio_return(investment, simulation.cash_out())) * 100)
    return daily_returns, value

"""Get predictions for a single symbol"""
def predict_for_symbol(symbols, start_date, end_date, clf_name="Linear Regression"):
    window = 5
    symbol = symbols[0]
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    start_date = start_date - timedelta(days=window)
    end_date = end_date + timedelta(days=window)
    
    prices_df, prices_df_with_spy = get_prices(symbols, start_date, end_date)
    ticker = TickerAnalysed(symbol=symbol, data_df=prices_df[[symbol]])
    # analyze_features(ticker)        
    prediction_df = predict(ticker, clf_name)
    plot_data(prediction_df, title=symbol + " Prediction vs actual")
    return prediction_df[window-2:], ticker.get_adj_close_df()[window-2:]
        
"""Train and test a set of symbols using a number of different classifier then return the model performance"""
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

"""Visualize the model performance"""
def visualize_classifier_results():
    df = get_classifier_analysis()
    df.drop(df.columns[0], axis=1, inplace=True)
    symbols = df['Symbol']
    df = df[df['Symbol'] == symbols[0]]
    df = df[df['Classifier'] != 'SVM Regression Poly']
    df.drop('Symbol', axis=1, inplace=True)
    df.set_index('Classifier', inplace=True)
    df.sort_values(by=['RMSE Out of Sample'], inplace=True)
    df_rmse = df[['RMSE Out of Sample']]
    print df_rmse
    
    ax = df_rmse.plot(kind='bar', title ="Classifier Comparison", figsize=(15, 10), legend=True, fontsize=10)
    ax.set_xlabel("Classifier", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    plt.xticks(rotation='horizontal')
    plt.show()

"""Default method for this module"""
def run(): 
    # make sure that you only use dates when the S&P 500 was trading
    train_start ='2017-01-02'
    train_end = ' 2017-12-01'
    # Buy date and sell date must within the same week and do not include holidays or weekends
    buy_date = '2017-12-04' 
    sell_date = '2017-12-08'
    investment = 10000 # $10,000.00 as starting investment
    # get all ticker symbols of the S&P 500
    symbols = get_all_tickers()
    # use equal weights for all of the included symbols
    weights = get_allocations_used(symbols, [])

    # optimize the S&P 500 portfolio
    print '----------------------------------------'
    print 'Generating S&P 500 optimized portfolio..'
    print '----------------------------------------'
    # generate an optimized portfolio using the S&P 500 based on the training dates mentioned above
    # store_optimized_portfolio(generate_optimized_portfolio(symbols, train_start, train_end, investment))
    optim_port = get_optimized_portfolio()
    
    # retrieve optimized portfolio in case you dont want to run the long optimization process
    print '----------------------------------------'
    print 'Training various models..'
    print '----------------------------------------'
    # train and test the optimized portfolio data against a number of regression algorithms 
    df = predict_for_symbols(optim_port.symbols, train_start, train_end)
    # save the models generated by the previous step
    store_classifer_analysis(df)  
    # visualize a comaparison of the different classifiers
    visualize_classifier_results() 
   
    print 'optim port: ', optim_port
    print '----------------------------------------'
    print 'Optimized portfolio symbols: ', optim_port.symbols
    print '----------------------------------------'
    print '----------------------------------------'
    print 'Optimized portfolio weights: ', optim_port.weights
    print '----------------------------------------'

    print '----------------------------------------'
    print 'Generating Performance information..'
    print '----------------------------------------'

    # test the portfolio performance in different scenarios  
    daily_returns_df = pd.DataFrame(index=pd.date_range(buy_date, sell_date))
    value_df = pd.DataFrame(index=pd.date_range(buy_date, sell_date))

    # Generate the S&P performance data during the buy and sell date range
    spy_daily_returns, spy_value = hold_spy(investment, buy_date, sell_date)
    daily_returns_df['SPY'] = spy_daily_returns
    value_df['SPY'] = spy_value

    # Generate the optimized portfolio performance data during the buy and sell date range without using predictions or active trading
    portfolio_hold_daily_returns, hold_value =  hold_optimized_portfolio(investment, buy_date, sell_date, optim_port.symbols, optim_port.weights)
    daily_returns_df['Hold'] = portfolio_hold_daily_returns
    value_df['hold'] = hold_value
    
    # Actively trade the optimized portfolio using the linear regression model
    portfolio_prediction_daily_returns, predict_value = use_predictions_optimized_portfolio(investment, buy_date, sell_date, train_start, train_end, optim_port.symbols)
    daily_returns_df['Predict'] = portfolio_prediction_daily_returns
    value_df['predict'] = predict_value

    # plot the results for comparison
    plot_data(daily_returns_df, 'Daily Return Comparison', 'Date', 'Return')
    plot_data(value_df, 'Portfolio Performance Comparison')


if __name__ == "__main__":
    run()