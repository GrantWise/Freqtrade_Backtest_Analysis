# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 19:58:48 2021

@author: gwise
"""
import datetime as dt
import json
import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
# import ffn

from datetime import date
# from binance.client import Client
# package for flattening json in pandas df
from pandas.io.json import json_normalize
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier

from numpy.random import *

# Define Constants
# client = Client('', '')
end = date.today().strftime("%Y.%m.%d")
start = '2021.05.18'
quote = 'USDT'
ticker = 'BTC'
interval = '1d'
top_n = 5
min_trade_ratio = 0.02

# =============================================================================
# # Load OHLCV Data for Benchmark
# ohlcv = client.get_historical_klines(symbol = ticker + quote, interval= interval, start_str= start)
# df_ticker = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number_trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
# df_ticker.drop(columns = ['Open', 'High', 'Low', 'Volume', 'Number_trades', 'Close time','Quote asset volume','Taker buy base asset volume','Taker buy quote asset volume','Ignore'], inplace = True)
# df_ticker[['Close']] = df_ticker[['Close']].astype(float)
# df_ticker['Date'] = pd.to_datetime(df_ticker['Date'], unit='ms')
#
# df = df_ticker
# df.set_index("Date", inplace = True)
# df['BTC'] = df.pct_change()
# df['cum_return'] = df['BTC'].cumsum()
# =============================================================================

# Backtest results path
#backtest_results_path = '/home/hippocrite/freqtrade//docker/user_data/backtest_results/'
backtest_results_path = 'C://ft_userdata//user_data//backtest_results'


# Load Backtest Results, check path in line 100
backtest_results = [

    "NFI-result-20210519-20210728_down"
]

# Columns to drop from trades dictionary,
# can be replaced by using meta in normalize function
drop_columns = ["stake_amount",
                "amount",
                "close_date",
                "open_rate",
                "close_rate",
                "fee_open",
                "fee_close",
                "profit_abs",
                "initial_stop_loss_abs",
                "initial_stop_loss_ratio",
                "stop_loss_abs",
                "stop_loss_ratio",
                "min_rate",
                "max_rate",
                "is_open",
                "open_timestamp",
                "close_timestamp"]


# Get Dictionary Keys
def getList(dict):
    return list(dict.keys())


# Function to convert string to datetime
def convert(date_time):
    format = "%Y.%m.%d"  # The format
    datetime_str = dt.datetime.strptime(date_time, format)

    return datetime_str


# Plots Horizontal Bar Chart
def bar_h_plot(df, data, title, x_label, y_label, first_date, last_date):
    '''
    Horizontal Bar Plot
    df      is a dataframe where the index is plotted on the Y-Axis
    data    is the dataframe column name to be plotted
    title   is a string used for the figure title
    x_label label for the X-axis
    y_label label for the Y-axis

    first_date and last_date are appended and plotted as a test box
    to show the date range of the back test results.
    '''
    fig, ax = plt.subplots(1, 1, figsize=(15, 7))

    plt.barh(df.index,
             df[data])

    ax.axvline(0.01, color='#fc4f30')
    ax.axvline(df[data].median(),
               color='orange',
               linestyle='--',
               label='Median')

    ax.axvline(df[data].mean(),
               color='black',
               linestyle='--',
               label='Mean')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label)

    ax.text(0.92, 0.15, first_date + '-' + last_date,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            bbox=dict(edgecolor='black', facecolor='white'))

    plt.xlim([0.8 * df[data].min(),
              1.05 * df[data].max()])
    plt.legend(loc='lower right')
    plt.tight_layout()
    ax.grid(axis='x')
    plt.show()
    # fig.savefig('AverageTradeProfitRatio.png')

    return


# Line plot for multiple strategies
def line_plot(strategies, df, num):
    '''
    Daily Line Plot of strategies defined in list
    sorted in ascending order.
    df is the data to be graphed
    Graphs bottom n strategies from list
    '''
    fig, ax = plt.subplots(nrows=num, ncols=1, figsize=(
        15, 12), sharex=True, sharey=True)

    n = 0
    for strategy in strategies[-num:]:
        ax[n].plot(df[strategy])

        ax[n].axhline(df[strategy].mean(), color='cyan',
                      linestyle='--', linewidth=1, label='Mean')

        ax[n].axhline(0,
                      color='red', linestyle='--', linewidth=1)

        ax[n].set_title(strategy, loc='left', fontsize=10)
        n += 1

    fig.suptitle('Daily Strategy Returns (%)', fontsize=16)
    fig.supylabel('Return (%)', fontsize=16)
    plt.tight_layout()

    plt.show()
    # fig.savefig('DailyReturns.png')

    return


# Read the first file to get a list of the strategies in the first file,
# assume all files contain the same strategies
jsonPath = os.path.join(backtest_results_path, backtest_results[0] + ".json")
print(jsonPath)
with open(jsonPath) as f:
    data = json.load(f)
strategies = getList(data["strategy"])
df_daily_returns = pd.DataFrame(columns=strategies)

# =============================================================================
# Testing Data
# file = 'backtest-result-20210728-20210817_up'
# trade = 0
# strategy = 'BigZ03'
# =============================================================================

# Loop over all the files in the backtest_results list
for file in backtest_results:
    print(file)
    jsonFile = os.path.join(backtest_results_path, file + '.json')

    # Open Json File
    with open(jsonFile) as f:
        data = json.load(f)

    s = data["strategy"]
    df_trade_profit = pd.DataFrame(
        columns=(
            'pair',
            'open_date',
            'trade_duration',
            'profit_ratio',
            'sell_reason',
            'strategy'))
    df_daily_profit = pd.DataFrame(
        columns=(
            'date',
            'daily_profit',
            'strategy'))

    # Loop over all strategies in strategies list and normalize
    for strategy in strategies:
        # print(strategy)

        # Populate Trade Performance Dataframe
        # s[strategy]['trades']
        df_temp = pd.json_normalize(s[strategy],
                                    record_path='trades')  # Figure out how to drop columns using path and meta
        df_temp['strategy'] = strategy
        df_temp.drop(drop_columns, axis=1, inplace=True)
        df_trade_profit = df_trade_profit.append(df_temp)

        # Populate Daily Returns Dataframe
        # s[strategy]['daily_profit']
        df_temp_daily = pd.json_normalize(
            s[strategy], record_path='daily_profit')

        df_temp_daily.rename(
            columns={0: 'date', 1: 'daily_profit'}, inplace=True)

        df_temp_daily['strategy'] = strategy
        df_temp_daily["date"] = pd.to_datetime(df_temp_daily['date'])
        df_daily_profit = df_daily_profit.append(df_temp_daily)
        df_daily_profit = df_daily_profit.iloc[:-1]

    df_daily_profit = df_daily_profit.pivot(
        index="date", columns="strategy", values="daily_profit")

    df_daily_profit.fillna(0, inplace=True)
    df_trade_profit.reset_index(drop=True, inplace=True)
    df_daily_returns = df_daily_returns.append(df_daily_profit)

# Slice Data to Start Date for analysis
df_s = df_daily_returns.loc[convert(start):]

df_cum_returns = df_s.cumsum()
df_daily_profit = df_s / 1000  # Initial stake amount 1000

df = df_trade_profit.groupby('strategy').filter(lambda x:
                                                (
                                                    (x['profit_ratio'].mean() > min_trade_ratio) &
                                                    (x['profit_ratio'].median()
                                                     > min_trade_ratio)
                                                ))

# Calculate the mean trade ratio for all strategies
strategy_mean_trade_ratio = df_trade_profit.groupby(
    'strategy').mean('profit_ratio').sort_values('profit_ratio')
# strategy_mean_trade_ratio = df_trade_profit.groupby('strategy').mean('profit_ratio') # No sort

# First and Last day of trades to be charted
first_date = df_daily_returns.first_valid_index().strftime("%Y.%m.%d")
last_date = df_daily_returns.last_valid_index().strftime("%Y.%m.%d")
period = (df_daily_returns.last_valid_index() -
          df_daily_returns.first_valid_index()).days
period_y = period / 365.2425

top_strategies = strategy_mean_trade_ratio.index.tolist()

top_cum_returns = df_cum_returns.tail(1)
top_cum_returns.reset_index(drop=True, inplace=True)
top_cum_returns = top_cum_returns.transpose()
top_cum_returns.columns = ['cum_ret']
top_cum_returns.sort_values("cum_ret", axis=0, ascending=True, inplace=True)
top_cum_ret_strategies = strategy_mean_trade_ratio.index.tolist()

cdgr = ((((df_cum_returns.iloc[-1]+1000) /
        (df_cum_returns.iloc[0]+1000)) ** (1 / period)) - 1).to_frame()
cdgr.columns = ['cdgr']
cdgr.sort_values("cdgr", axis=0, ascending=True, inplace=True)

top_strategies[-top_n:]
top_cum_returns[-top_n:]

# Save Daily profit dataframe to csv
# df_daily_returns.to_csv('file_name.csv')
# strategy_mean_trade_ratio.to_csv('returns.csv')

# Plots

# Bar Plot of Average Trade Profit Ratio
bar_h_plot(strategy_mean_trade_ratio,
           'profit_ratio',
           'Average Trade Profit Ratio',
           'Trade Ratio',
           'Strategy',
           first_date,
           last_date)

# Bar Plot of Cumulative Return
bar_h_plot(top_cum_returns,
           'cum_ret',
           'Cumulative Return by Strategy',
           'Return',
           'Strategy',
           first_date,
           last_date)

# Bar Plot of Compound Daily Growth Rate
bar_h_plot(cdgr,
           'cdgr',
           'Compound Daily Growth Rate by Strategy',
           'CDGR (%)',
           'Strategy',
           first_date,
           last_date)


# Violin Plot of Strategies
fig, ax = plt.subplots(1, 1, figsize=(15, 7))
sns.violinplot(y=df['strategy'], x=df['profit_ratio'], color="skyblue")

ax.axvline(0, color='#fc4f30')

ax.set_title('Strategies where Mean & Median Trade Ratio > ' + str(min_trade_ratio),
             fontsize=16)

plt.show()
# fig.savefig('AverageTradeProfitRatioVP.png')


# Plot 1 Cumulative Returns
fig, ax = plt.subplots(1, 1, figsize=(14, 7))

df_cum_returns[top_strategies[-top_n:]].plot(ax=ax)
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(
    [df_cum_returns[strategy].iloc[-1]
     for strategy in df_cum_returns[top_strategies[-top_n:]].columns])

ax2.set_yticklabels(df_cum_returns[top_strategies[-top_n:]].columns)

ax.set_title('Cumulative Strategy Returns')
ax.set_ylabel('Returns (USD)',)

plt.tight_layout()
ax.legend(fontsize=10, loc='upper left')
ax.grid(False)

plt.show()
# fig.savefig('CumulativeDailyReturns.png')

for strategy in top_strategies[-top_n:]:
    ax.plot(df_cum_returns[strategy], label=strategy)


# Plot 2
# Time series plot of daily returns per strategy

a = ['NostalgiaForInfinityV7_10_1', 'NostalgiaForInfinityV7_10_0',
     'NostalgiaForInfinityV7', 'NostalgiaForInfinityV6', 'NostalgiaForInfinityV7_2_0']
line_plot(top_strategies, df_daily_profit, top_n)
line_plot(a, df_daily_profit, top_n)


# Plot 3
# Histograms of Daily Returns per strategy

# Calculate subplot grid
ncolumns = 3
nrows = math.ceil(top_n/ncolumns)


fig, ax = plt.subplots(nrows, ncolumns, figsize=(15, 7),
                       sharex=True, sharey=True)
print(ax)
bins = [-0.1, -0.0875, -0.075, -0.0625, -0.05,
        -0.0375, -0.025, -0.0125, 0, 0.0125,
        0.025, 0.0375, 0.05, 0.0625, 0.075,
        0.0875, 0.1, 0.1125, 0.125, 0.1375, 0.15]
n = 0
for strategy in top_strategies[-top_n:]:
    ax[n, 0].hist(df_daily_profit[strategy],
                  bins=bins,
                  density=True,
                  edgecolor='b',
                  label=strategy)

    ax[n, 0].set_title(strategy,
                       fontsize=10)

    ax[n, 0].axvline(0,
                     color='#fc4f30')

    ax[n, 0].axvline(df_daily_profit[strategy].mean(),
                     color='orange',
                     linestyle='--')

    n += 1

fig.suptitle('Daily Strategy Returns 2021-01-01 : 2021-08-15 (%)', fontsize=16)
fig.supylabel('Return (%)', fontsize=16)
plt.tight_layout()


plt.show()
# fig.savefig('HistDailyReturns.png')


# Trade Anlaysis


# Strategy Portfolio

# Use above analysis to exclude strategies before running this, currently the weights will not make much sense.
# This is still very much work in progress and needs a lot of work and I am not sure if it is a theoretically sound application of the method

# Calculate expected returns and sample covariance
mu = expected_returns.ema_historical_return(df_daily_profit,
                                            returns_data=True,
                                            compounding=True,
                                            span=120,
                                            frequency=365)

mu.sort_values().plot.barh(figsize=(10, 6))

# S = risk_models.exp_cov(df_daily_profit, returns_data=True, span=180, frequency=365, log_returns=False)

S = risk_models.CovarianceShrinkage(
    df_daily_profit, returns_data=True).ledoit_wolf()

plotting.plot_covariance(S, plot_correlation=True)

# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)

cleaned_weights = ef.clean_weights()
ef.save_weights_to_file("weights.txt")  # saves to file
print(cleaned_weights)

# Plot of portfolio weights

# Not working as I need to figure out how to convert dictionary to dataframe

# =============================================================================
# cleaned_weights.keys()
#
# pd.DataFrame(orderedDictList, columns=orderedDictList.keys())
#
# fig, ax = plt.subplots(1,1, figsize = (15,12))
#
# ax[0,0].barh(cleaned_weights)
#
# ax.set_title('Cumulative Strategy Returns')
# ax.set_ylabel('Returns (USD)')
# plt.tight_layout()
# ax.legend(fontsize=10)
# ax.grid(False)
# plt.show()
# =============================================================================

#
# pd.DataFrame(orderedDictList, columns=orderedDictList.keys())
#
# fig, ax = plt.subplots(1,1, figsize = (15,12))
#
# ax[0,0].barh(cleaned_weights)
#
# ax.set_title('Cumulative Strategy Returns')
# ax.set_ylabel('Returns (USD)')
# plt.tight_layout()
# ax.legend(fontsize=10)
# ax.grid(False)
# plt.show()
# =============================================================================
()
# =============================================================================

#
# pd.DataFrame(orderedDictList, columns=orderedDictList.keys())
#
# fig, ax = plt.subplots(1,1, figsize = (15,12))
#
# ax[0,0].barh(cleaned_weights)
#
# ax.set_title('Cumulative Strategy Returns')
# ax.set_ylabel('Returns (USD)')
# plt.tight_layout()
# ax.legend(fontsize=10)
# ax.grid(False)
# plt.show()
# =============================================================================