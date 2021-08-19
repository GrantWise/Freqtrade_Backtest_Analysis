# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 19:58:48 2021

@author: gwise
"""

import json
import pandas as pd
import numpy as np
import datetime as dt

from pandas.io.json import json_normalize #package for flattening json in pandas df

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting

from binance.client import Client
from datetime import date

import matplotlib.pyplot as plt
import seaborn as sns

# Define Constants
client = Client('','')
end = date.today().strftime("%Y.%m.%d")
start = '2021.01.01'
quote = 'USDT'
ticker = 'BTC'
interval = '1d'

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


# Load Backtest Results, check path in line 100
backtest_results = [
    "backtest-result-20180115-20180420_down",
    "backtest-result-20180420-20180518_up",
    "backtest-result-20180518-20180724_down",
    "backtest-result-20180724-20181112_flat",
    "backtest-result-20181112-20190218_down",
    "backtest-result-20190218-20190717_up",
    "backtest-result-20190717-20200107_down",
    "backtest-result-20200107-20200229_up",
    "backtest-result-20200229-20200422_down",
    "backtest-result-20200422-20200906_up",
    "backtest-result-20200906-20201009_flat",
    "backtest-result-20201009-20210128_up",
    "backtest-result-20210128-20210324_up",
    "backtest-result-20210324-20210519_up",
    "backtest-result-20210519-20210728_down",
    "backtest-result-20210728-20210818_up"
    ]

# Columns to drop from trades dictionary, can be replaced by using meta in normalize function
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
    format = "%Y.%m.%d" # The format
    datetime_str = dt.datetime.strptime(date_time, format)

    return datetime_str

# Read the first file to get a list of the strategies in the first file, assume all files contain the same strategies
with open("C:\\ft_userdata\\user_data\\backtest_results\\" + backtest_results[0] + ".json") as f:
    data = json.load(f)
strategies = getList(data["strategy"])
df_daily_returns = pd.DataFrame(columns = strategies)

# =============================================================================
# Testing Data
# file = 'backtest-result-20210728-20210817_up'
# trade = 0
# strategy = 'BigZ03'
# =============================================================================

# Loop over all the files in the backtest_results list
for file in backtest_results:
    print(file)
    
    # Open Json File
    with open("C:\\ft_userdata\\user_data\\backtest_results\\" + file + ".json") as f:
        data = json.load(f)
        
    s = data["strategy"]
    df_trade_profit = pd.DataFrame(columns=('pair', 'open_date', 'trade_duration', 'profit_ratio', 'sell_reason', 'strategy'))
    df_daily_profit = pd.DataFrame(columns=('date', 'daily_profit', 'strategy'))
    
    # Loop over all strategies in strategies list and normalize
    for strategy  in strategies:
        # print(strategy)
        
        # Populate Trade Performance Dataframe
        # s[strategy]['trades']
        df_temp = json_normalize(s[strategy], record_path='trades') # Figure out how to drop columns using path and meta
        df_temp['strategy'] = strategy
        df_temp.drop(drop_columns, axis = 1, inplace = True) 
        df_trade_profit = df_trade_profit.append(df_temp)
        
        # Populate Daily Returns Dataframe
        # s[strategy]['daily_profit']
        df_temp_daily  = json_normalize(s[strategy], record_path='daily_profit')
        df_temp_daily.rename(columns = {0:'date', 1:'daily_profit'}, inplace = True)
        df_temp_daily['strategy'] = strategy
        df_temp_daily["date"] = pd.to_datetime(df_temp_daily['date'])
        df_daily_profit = df_daily_profit.append(df_temp_daily)
        df_daily_profit = df_daily_profit.iloc[:-1]

    df_daily_profit = df_daily_profit.pivot(index = "date", columns = "strategy", values = "daily_profit")
    df_daily_profit.fillna(0, inplace = True)
    df_trade_profit.reset_index(drop=True, inplace=True)
    df_daily_returns = df_daily_returns.append(df_daily_profit)

# Slice Data to Start Date for analysis
df_s = df_daily_returns.loc[convert(start):]

df_cum_returns = df_s.cumsum() 
df_daily_profit = df_s / 1000 # Initial stake amount 1000


df = df_trade_profit.groupby('strategy').filter(lambda x : 
                                                (
                                                    (x['profit_ratio'].mean() > 0.0125) & 
                                                    (x['profit_ratio'].median() > 0.0125)
                                                ))


# Plots

# Bar Plot of Average Trade Profit Ratio

fig, ax = plt.subplots(1,1, figsize = (15,7))

plt.barh(df_trade_profit.groupby('strategy').mean('profit_ratio').index, 
        df_trade_profit.groupby('strategy').mean('profit_ratio')['profit_ratio'])

ax.axvline(0.01, color='#fc4f30')
ax.axvline(0.0125, color='orange', linestyle='--', label='0.0125')

ax.set_title('Average Trade Profit Ratio by Strategy', fontsize=16)
ax.set_xlabel('Trade Ratio', fontsize=14)
ax.set_ylabel('Strategy')
ax.text(0.95, 0.05, start + '-', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
        bbox=dict(edgecolor='black', facecolor='white'))

plt.tight_layout()
ax.grid(axis='x')

plt.show()
fig.savefig('AverageTradeProfitRatio.png')

# Violin Plot of Strategies

fig, ax = plt.subplots(1,1, figsize = (15,7))

sns.violinplot(y=df['strategy'], x=df['profit_ratio'], color="skyblue")
# sns.boxplot(y=df['strategy'], x=df['profit_ratio'])

# sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)

plt.show()
fig.savefig('AverageTradeProfitRatioVP.png')


# Bar Plot of Strategies Mean Trade ratio

fig, ax = plt.subplots(1,1, figsize = (15,7))

plt.barh(y=df_trade_profit['strategy'], width=df_trade_profit['profit_ratio'].mean())

plt.show()




# Strategies to plot
## Need to make this a dynamic list that changes based on specific criteria e.g mean, trades/day etc
g1 = 'ElliotV5'
g4 = 'NostalgiaForInfinityV7'
g2 = 'NotAnotherSMAOffsetStrategy'
g6 = 'NostalgiaForInfinityV6'
g3 = 'Combined_NFIv7_SMA'
g5 = 'NostalgiaForInfinityV8'
# g6 = 'Obelisk_Ichimoku_Slow_v1_3'

# Plot 1: Cumulative Returns since start date

fig, ax = plt.subplots(1,1, figsize = (14,7))

## Need to iterate over strategies
ax.plot(df_cum_returns[g1], label = g1 )
ax.plot(df_cum_returns[g2], label = g2 )
ax.plot(df_cum_returns[g3], label = g3 )
ax.plot(df_cum_returns[g4], label = g4 )
ax.plot(df_cum_returns[g5], label = 'NostalgiaForInfinityV7.6.1' )
ax.plot(df_cum_returns[g6], label = g6 )

ax.axvline(convert('2021.01.27'), color='red', linestyle='--', linewidth=1)
ax.axvline(convert('2021.02.22'), color='green', linestyle='--', linewidth=1)
ax.axvline(convert('2021.02.28'), color='red', linestyle='--', linewidth=1)
ax.axvline(convert('2021.03.21'), color='green', linestyle='--', linewidth=1)
ax.axvline(convert('2021.03.25'), color='red', linestyle='--', linewidth=1)
ax.axvline(convert('2021.04.17'), color='green', linestyle='--', linewidth=1)
ax.axvline(convert('2021.04.25'), color='red', linestyle='--', linewidth=1)
ax.axvline(convert('2021.05.11'), color='green', linestyle='--', linewidth=1)
ax.axvline(convert('2021.06.26'), color='red', linestyle='--', linewidth=1)
ax.axvline(convert('2021.07.20'), color='red', linestyle='--', linewidth=1)

ax.set_title('Cumulative Strategy Returns')
ax.set_ylabel('Returns (USD)')
plt.tight_layout()
ax.legend(fontsize=10)
ax.grid(False)
plt.show()
fig.savefig('CumulativeDailyReturns.png')

# Plot 2
# Time series plot of daily returns per strategy
fig, ax = plt.subplots(6,1, figsize = (15,12), sharex=True)

# Need to figure out how to iterate over axes
# Plot 1
ax[0].plot(df_daily_profit[g1])
ax[0].axhline(df_daily_profit[g1].mean(), color='cyan', linestyle='--', linewidth=1, label='Mean')
ax[0].axhline(0, color='red', linestyle='--', linewidth=1, label='Mean')
ax[0].set_title(g1, loc='left', fontsize=10)

# Plot 2
ax[1].plot(df_daily_profit[g2])
ax[1].axhline(df_daily_profit[g2].mean(), color='cyan', linestyle='--', linewidth=1, label='Mean')
ax[1].axhline(0, color='red', linestyle='--', linewidth=1, label='Mean')
ax[1].set_title(g2, loc='left', fontsize=10)

# Plot 3
ax[2].plot(df_daily_profit[g3])
ax[2].axhline(df_daily_profit[g3].mean(), color='cyan', linestyle='--', linewidth=1, label='Mean')
ax[2].axhline(0, color='red', linestyle='--', linewidth=1, label='Mean')
ax[2].set_title(g3, loc='left', fontsize=10)

# Plot 4
ax[3].plot(df_daily_profit[g4])
ax[3].axhline(df_daily_profit[g4].mean(), color='cyan', linestyle='--', linewidth=1, label='Mean')
ax[3].axhline(0, color='red', linestyle='--', linewidth=1, label='Mean')
ax[3].set_title(g4, loc='left', fontsize=10)

# Plot 5
ax[4].plot(df_daily_profit[g5])
ax[4].axhline(df_daily_profit[g5].mean(), color='cyan', linestyle='--', linewidth=1, label='Mean')
ax[4].axhline(0, color='red', linestyle='--', linewidth=1, label='Mean')
ax[4].set_title(g5, loc='left', fontsize=10)

# Plot 6
ax[5].plot(df_daily_profit[g6])
ax[5].axhline(df_daily_profit[g6].mean(), color='cyan', linestyle='--', linewidth=1, label='Mean')
ax[5].axhline(0, color='red', linestyle='--', linewidth=1, label='Mean')
ax[5].set_title(g6, loc='left', fontsize=10)

fig.suptitle('Daily Strategy Returns (%)', fontsize=16)
fig.supylabel('Return (%)', fontsize=16)
plt.tight_layout()

plt.show()
# fig.savefig('DailyReturns.png')

# Plot 3
# Histograms of Daily Returns per strategy

fig, ax = plt.subplots(2,3, figsize = (15,7), sharex=True, sharey=True)

bins = [-0.1,-0.0875, -0.075, -0.0625, -0.05,
        -0.0375, -0.025, -0.0125, 0, 0.0125,
        0.025, 0.0375, 0.05, 0.0625, 0.075,
        0.0875, 0.1, 0.1125, 0.125, 0.1375, 0.15]

# Plot 1
ax[0,0].hist(df_daily_profit[g1],bins=bins, density=True, edgecolor='b', label=g1)
ax[0,0].set_title(g1, fontsize=10)
ax[0,0].axvline(0, color='#fc4f30')
ax[0,0].axvline(df_daily_profit[g1].mean(), color='orange', linestyle='--')

# Plot 2
ax[0,1].hist(df_daily_profit[g2],bins = bins, density = True, edgecolor = 'b', label = g2)
ax[0,1].set_title(g2, fontsize=10)
ax[0,1].axvline(0, color='#fc4f30')
ax[0,1].axvline(df_daily_profit[g2].mean(), color='orange', linestyle='--')

# Plot 3
ax[0,2].hist(df_daily_profit[g3],bins = bins, density = True, edgecolor = 'b', label = g3)
ax[0,2].set_title(g3, fontsize=10)
ax[0,2].axvline(0, color='#fc4f30')
ax[0,2].axvline(df_daily_profit[g3].mean(), color='orange', linestyle='--')

# Plot 4
ax[1,0].hist(df_daily_profit[g4],bins = bins, density = True, edgecolor = 'b', label = g4)
ax[1,0].set_title(g4, fontsize=10)
ax[1,0].axvline(0, color='#fc4f30')
ax[1,0].axvline(df_daily_profit[g4].mean(), color='orange', linestyle='--')

# Plot 5
ax[1,1].hist(df_daily_profit[g5],bins = bins, density = True, edgecolor = 'b', label = g5)
ax[1,1].set_title('NostalgiaForInfinityV7.6.1', fontsize=10)
ax[1,1].axvline(0, color='#fc4f30')
ax[1,1].axvline(df_daily_profit[g5].mean(), color='orange', linestyle='--')

# Plot 6
ax[1,2].hist(df_daily_profit[g6],bins = bins, density = True, edgecolor = 'b', label = g6)
ax[1,2].set_title(g6, fontsize=10)
ax[1,2].axvline(0, color='#fc4f30')
ax[1,2].axvline(df_daily_profit[g6].mean(), color='orange', linestyle='--')

fig.suptitle('Daily Strategy Returns 2021-01-01 : 2021-08-15 (%)', fontsize=16)
fig.supylabel('Return (%)', fontsize=16)
plt.tight_layout()

plt.show()
# fig.savefig('HistDailyReturns.png')

# Trade Anlaysis

df = df_trade_profit.groupby('strategy').filter(lambda x : 
                                                (
                                                    (x['profit_ratio'].mean() > 0.01) & 
                                                    (x['profit_ratio'].median() > 0.01)
                                                ))


df_trade_strategy = df_trade_profit.groupby('strategy').mean('profit_ratio')











# Strategy Portfolio

# Use above analysis to exclude strategies before running this, currently the weights will not make much sense.
# This is still very much work in progress and needs a lot of work and I am not sure if it is a theoretically sound application of the method

# Calculate expected returns and sample covariance
mu = expected_returns.ema_historical_return(df_daily_profit, returns_data=True, compounding=True, span=120, frequency=365)
mu.sort_values().plot.barh(figsize=(10,6))

# Calculate expected returns and sample covariance
mu = expected_returns.ema_historical_return(df_daily_profit, returns_data=True, compounding=True, span=180, frequency=365)
# mu = expected_returns.capm_return(df_daily_profit, returns_data=True)
mu.plot.barh(figsize=(10,6))

# S = risk_models.exp_cov(df_daily_profit, returns_data=True, span=180, frequency=365, log_returns=False) 
S = risk_models.CovarianceShrinkage(df_daily_profit, returns_data=True).ledoit_wolf()
plotting.plot_covariance(S, plot_correlation=True);


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
