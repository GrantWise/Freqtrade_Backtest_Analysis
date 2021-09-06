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

from datetime import date
from matplotlib.dates import DateFormatter
from datetime import datetime

from binance.client import Client
# package for flattening json in pandas df
from pandas.io.json import json_normalize
from pypfopt import expected_returns
from pypfopt import plotting
from pypfopt import risk_models
from pypfopt.efficient_frontier import EfficientFrontier

from numpy.random import *

# Define Constants
client = Client('', '')
date_end = date.today().strftime("%Y.%m.%d")
date_start = '2021.05.18'
quote = 'USDT'
ticker = 'BTC'
interval = '1d'
top_n = 6
min_trade_ratio = 0.02


# Backtest results path
# backtest_results_path = '/home/hippocrite/freqtrade//docker/user_data/backtest_results/'
backtest_results_path = 'C://ft_userdata//user_data//backtest_results'


# Load Backtest Results, check path in line 100
backtest_results = [

    "NFI-result-20210728-20210905_up"
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


def month_list(df):
    '''
    Pass a dataframe where the index is a time series
    return list of months
    '''
    period = df.index.tolist()
    start = period[0]
    end = period[-1]
    month_list = [i.strftime("%d-%b-%y") for i in pd.date_range(start, end, freq='MS')]
    return month_list


def benchmark(ticker, quote=quote, start=date_start, end=date_end, interval='1d'):
    '''
    Load OHLCV Data for Benchmark
    '''
    ohlcv = client.get_historical_klines(
        symbol=ticker + quote, interval=interval, start_str=start, end_str=end)
    df_ticker = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                             'Quote asset volume', 'Number_trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df_ticker.drop(columns=['Open', 'High', 'Low', 'Volume', 'Number_trades', 'Close time', 'Quote asset volume',
                   'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], inplace=True)
    df_ticker[['Close']] = df_ticker[['Close']].astype(float)
    df_ticker['Date'] = pd.to_datetime(df_ticker['Date'], unit='ms')

    df = df_ticker
    df.set_index("Date", inplace=True)
    df['BTC'] = df.pct_change()
    df['cum_return'] = df['BTC'].cumsum()
    return(df)


# Plots Horizontal Bar Chart
def bar_h_plot(df,
               data,
               title="Title",
               x_label="X-Label",
               y_label="Y-Label",
               first_date=date_start,
               last_date=date_end):
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
def line_plot(strategies,
              df,
              num,
              title="Title",
              x_label="X-Label",
              y_label="Y-Label",
              first_date=date_start,
              last_date=date_end):
    '''
    Daily Line Plot of strategies defined in list
    sorted in ascending order.
    df is the data to be graphed
    Graphs bottom n strategies from list
    '''
    months = month_list(df)

    fig, ax = plt.subplots(nrows=num, ncols=1, figsize=(
        15, 12), sharex=True, sharey=True)

    n = 0
    for strategy in strategies[-num:]:
        ax[n].plot(df[strategy])

        ax[n].axhline(df[strategy].mean(), color='cyan',
                      linestyle='--', linewidth=1, label='Mean')

        ax[n].axhline(0,
                      color='red', linestyle='--', linewidth=1)

        for i in months:
            ax[n].axvline(datetime.strptime(i, "%d-%b-%y"),
                          color='red', linestyle='--', linewidth=1)

        ax[n].set_title(strategy, loc='left', fontsize=10)
        n += 1

    fig.suptitle('Daily Strategy Returns (%)', fontsize=16)
    fig.supylabel('Return (%)', fontsize=16)
    plt.tight_layout()

    plt.show()
    # fig.savefig('DailyReturns.png')

    return


def get_strat_backtest():
    '''
    Read the first file to get a list of the strategies in the first file,
    assume all files contain the same strategies
    '''
    jsonPath = os.path.join(backtest_results_path, backtest_results[0] + ".json")
    print(jsonPath)
    with open(jsonPath) as f:
        data = json.load(f)
    strategies = getList(data["strategy"])
    return strategies


strategies = get_strat_backtest()

#######

# =============================================================================
# Testing Data
# file = 'NFI-result-20210728-20210905_up'
# trade = 0
# strategy = 'NostalgiaForInfinityV7'
# =============================================================================

df_daily_returns = pd.DataFrame(columns=strategies)

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

        # Populate Trade Performance Dataframe - JSON format s[strategy]['trades']
        # s[strategy]['trades']
        df_temp = pd.json_normalize(s[strategy],
                                    record_path='trades')  # Figure out how to drop columns using path and meta
        df_temp['strategy'] = strategy
        df_temp.drop(drop_columns, axis=1, inplace=True)
        df_trade_profit = df_trade_profit.append(df_temp)

        # Populate Daily Returns Dataframe - JSON format s[strategy]['daily_profit']
        df_temp_daily = pd.json_normalize(
            s[strategy], record_path='daily_profit')

        df_temp_daily.rename(
            columns={0: 'date', 1: 'daily_profit'}, inplace=True)

        df_temp_daily['strategy'] = strategy
        df_temp_daily["date"] = pd.to_datetime(df_temp_daily['date'])
        df_daily_profit = df_daily_profit.append(df_temp_daily)
        df_daily_profit = df_daily_profit.iloc[:-1]

    df_daily_profit = df_daily_profit.pivot(index="date",
                                            columns="strategy",
                                            values="daily_profit")

    df_daily_profit.fillna(0, inplace=True)
    df_trade_profit.reset_index(drop=True, inplace=True)
    df_daily_returns = df_daily_returns.append(df_daily_profit)

df_daily_profit = df_daily_returns / 1000  # Initial stake amount 1000


###############################################################################
# Dataframes constructed with reportable data
# df_daily_returns - Daily returns (Base Currency amount) for each strategy in the strategy list created from the first backtest file
# df_daily_profit - Daily Returns expressed as a percentage
# df_trade_profit - Profit ratio for each trade shows pair and strategy
###############################################################################


# Start Analysis


# Calculate Cumulative REturns from Start Date
# Slice Data to Start Date for analysis
df_s = df_daily_returns.loc[convert(date_start):]
df_cum_profits = df_s.cumsum()


# Filter Strategies that don't meet minimum trade ratio
df_min_trade_ratio = df_trade_profit.groupby('strategy').filter(lambda x:
                                                                (
                                                                    (x['profit_ratio'].mean() > min_trade_ratio) &
                                                                    (x['profit_ratio'].median()
                                                                     > min_trade_ratio)
                                                                ))

# Calculate the mean trade ratio for all strategies
strategy_mean_trade_ratio = df_trade_profit.groupby(
    'strategy').mean('profit_ratio').sort_values('profit_ratio')

# List of strategies sorted by trade ratio
top_strategies = strategy_mean_trade_ratio.index.tolist()


# First and Last day of trades to be charted
first_date = df_daily_returns.first_valid_index().strftime("%Y.%m.%d")
last_date = df_daily_returns.last_valid_index().strftime("%Y.%m.%d")

# Days betwwen first dtae and last date
period = (df_daily_returns.last_valid_index() -
          df_daily_returns.first_valid_index()).days
# Years between first date and last date
period_y = period / 365.2425


# Download Benchmark Data
df_bench = benchmark('BTC', start=first_date, end=last_date)


top_cum_returns = df_cum_profits.tail(1)
top_cum_returns.reset_index(drop=True, inplace=True)
top_cum_returns = top_cum_returns.transpose()
top_cum_returns.columns = ['cum_ret']
top_cum_returns.sort_values("cum_ret", axis=0, ascending=True, inplace=True)
top_cum_ret_strategies = strategy_mean_trade_ratio.index.tolist()

cdgr = ((((df_cum_profits.iloc[-1]+1000) /
        (df_cum_profits.iloc[0]+1000)) ** (1 / period)) - 1).to_frame()
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
sns.violinplot(y=df_min_trade_ratio['strategy'],
               x=df_min_trade_ratio['profit_ratio'], color="skyblue")

ax.axvline(0, color='#fc4f30')

ax.set_title('Strategies where Mean & Median Trade Ratio > ' + str(min_trade_ratio),
             fontsize=16)

plt.show()
# fig.savefig('AverageTradeProfitRatioVP.png')

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
sns.violinplot(y=df_trade_profit['strategy'], x=df_trade_profit['profit_ratio'], color="skyblue")

ax.axvline(0, color='#fc4f30')

ax.set_title('Strategies where Mean & Median Trade Ratio > ' + str(min_trade_ratio),
             fontsize=16)

plt.show()


# Plot 1 Cumulative Returns
months = month_list(df_cum_profits)

fig, ax = plt.subplots(1, 1, figsize=(14, 7))

df_cum_profits[top_strategies[-top_n:]].plot(ax=ax)
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(
    [df_cum_profits[strategy].iloc[-1]
     for strategy in df_cum_profits[top_strategies[-top_n:]].columns])

ax2.set_yticklabels(df_cum_profits[top_strategies[-top_n:]].columns)

for i in months:
    ax.axvline(datetime.strptime(i, "%d-%b-%y"), color='red', linestyle='--', linewidth=1)

ax.set_title('Cumulative Strategy Returns')
ax.set_ylabel('Returns (USD)',)

plt.tight_layout()
ax.legend(fontsize=10, loc='upper left')
ax.grid(False)

plt.show()
# fig.savefig('CumulativeDailyReturns.png')


# Plot 2
# Time series plot of daily returns per strategy

a = ['NostalgiaForInfinityV7_10_1', 'NostalgiaForInfinityV7_10_0', 'NostalgiaForInfinityV7_10_2', 'NostalgiaForInfinityV7_11_0',
     'NostalgiaForInfinityV7', 'NostalgiaForInfinityV6', 'NostalgiaForInfinityV7_3_1']

line_plot(top_strategies, df_daily_profit, top_n)
line_plot(a, df_daily_profit, top_n)


# Plot 3
# Histograms of Daily Returns per strategy

# Calculate subplot grid
ncolumns = 3
nrows = math.ceil(top_n/ncolumns)

# fig.savefig('HistDailyReturns.png')


fig, axes = plt.subplots(nrows, ncolumns, figsize=(16, 8), sharex=True)

for col, ax in zip(df_daily_profit.columns, axes.flatten()):
    ax.hist(df_daily_profit[col],
            density=True,
            edgecolor='b',
            bins=20,
            label=strategy)
    ax.axvline(0, color='#fc4f30')
    ax.axvline(df_daily_profit[strategy].mean(), color='orange', linestyle='--')
    ax.set_title(col)
    plt.subplots_adjust(wspace=.15, hspace=.5)
fig.suptitle('Daily Strategy Returns 2021-01-01 : 2021-08-15 (%)', fontsize=16)
fig.supylabel('Return (%)', fontsize=16)
plt.tight_layout()


plt.show()
# fig.savefig('HistDailyReturns.png')
            density=True,
            edgecolor='b',
            bins=20,
            label=strategy)
    ax.axvline(0, color='#fc4f30')
    ax.axvline(df_daily_profit[strategy].mean(), color='orange', linestyle='--')
    ax.set_title(col)
    plt.subplots_adjust(wspace=.15, hspace=.5)
fig.suptitle('Daily Strategy Returns 2021-01-01 : 2021-08-15 (%)', fontsize=16)
fig.supylabel('Return (%)', fontsize=16)
plt.tight_layout()


plt.show()
# fig.savefig('HistDailyReturns.png')
