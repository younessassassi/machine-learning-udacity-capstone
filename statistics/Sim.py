import pandas as pd

"""Trade simulation class"""

class Sim(object):
    
    def __init__(self, portfolio, start, end):
        self.portfolio = portfolio
        self.start = start
        self.end = end
        
        #set up the data frame for the stocks over the period
        self.daily_prices = portfolio_get_stock_prices()
        self.daily_prices['Investment'] = 1.0


        self.trades = self.daily_prices.copy()
        self.trades[0:] = 0.0
        self.trades.loc[0:, 'Cash'] = 10000

    

    # #step through the orders frame and for each order update the trades table
    # for index, row in orders_df.iterrows():
    #     symbol_to_update = row['Symbol']
    #     amount_to_update = row['Shares']

    #     if row['Order'] == "SELL":
    #         amount_to_update = amount_to_update * -1.0

    #     cash_gained_or_lost = amount_to_update * -1.0 * daily_prices_df.loc[index, symbol_to_update]
    #     current_holdings = amount_to_update + trades_df.loc[index, symbol_to_update]
    #     current_cash = cash_gained_or_lost + trades_df.loc[index, 'Cash']

    #     trades_df.loc[index:, symbol_to_update] = current_holdings
    #     trades_df.loc[index:, 'Cash'] = current_cash

    # value_of_portfolio = trades_df * daily_prices_df
    # portvals = value_of_portfolio.sum(axis=1)
    # if isinstance(portvals, pd.DataFrame):
    #     portvals = portvals[portvals.columns[0]]

    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # # Simulate a $SPX-only reference portfolio to get stats
    # prices_SPX = get_data(['$SPX'], dates)
    # prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    # portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    # cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # # Compare portfolio against $SPX
    # print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    # print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    # print
    # print "Cumulative Return of Fund: {}".format(cum_ret)
    # print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    # print
    # print "Standard Deviation of Fund: {}".format(std_daily_ret)
    # print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    # print
    # print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    # print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    # print
    # print "Final Portfolio Value: {}".format(portvals[-1])

    # # Plot computed daily portfolio value
    # df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    # plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")









# def calculate_portfolio_value(datafile, prices_all, dates, stock):
#      #read in the data from the order file
#     orders_df = pd.read_csv(datafile, index_col = 'Date', parse_dates = True)
#     orders_df = orders_df.sort_index(kind='mergesort')

#     #set up the data frame for the stocks over the period
#     daily_prices_df = pd.DataFrame(index=dates)
#     daily_prices_df = daily_prices_df.join(prices_all[stock])
#     daily_prices_df.dropna(inplace=True)
#     daily_prices_df['Cash'] = 1.0

#     #make an additional data frame called trades
#     trades_df = daily_prices_df.copy()
#     trades_df[0:] = 0.0
#     trades_df.loc[0:, 'Cash'] = 10000

#     #step through the orders frame and for each order update the trades table
#     for index, row in orders_df.iterrows():
#         symbol_to_update = row['Symbol']
#         amount_to_update = row['Shares']

#         if row['Order'] == "SELL":
#             amount_to_update = amount_to_update * -1.0

#         cash_gained_or_lost = amount_to_update * -1.0 * daily_prices_df.loc[index, symbol_to_update]
#         current_holdings = amount_to_update + trades_df.loc[index, symbol_to_update]
#         current_cash = cash_gained_or_lost + trades_df.loc[index, 'Cash']

#         trades_df.loc[index:, symbol_to_update] = current_holdings
#         trades_df.loc[index:, 'Cash'] = current_cash

#     value_of_portfolio = trades_df * daily_prices_df
#     portvals = value_of_portfolio.sum(axis=1)
#     if isinstance(portvals, pd.DataFrame):
#         portvals = portvals[portvals.columns[0]]

#     cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

#     # Simulate a $SPX-only reference portfolio to get stats
#     prices_SPX = get_data(['$SPX'], dates)
#     prices_SPX = prices_SPX[['$SPX']]  # remove SPY
#     portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
#     cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

#     # Compare portfolio against $SPX
#     print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
#     print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
#     print
#     print "Cumulative Return of Fund: {}".format(cum_ret)
#     print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
#     print
#     print "Standard Deviation of Fund: {}".format(std_daily_ret)
#     print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
#     print
#     print "Average Daily Return of Fund: {}".format(avg_daily_ret)
#     print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
#     print
#     print "Final Portfolio Value: {}".format(portvals[-1])

#     # Plot computed daily portfolio value
#     df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
#     plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")

