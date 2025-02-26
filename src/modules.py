#!usr/bin/env  python3

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import logging
import multiprocessing
from datetime import datetime, timedelta
import logging
import time

# Set yfinance logging level to ERROR to suppress DEBUG logs
logging.getLogger("yfinance").setLevel(logging.ERROR)

##############
# etf basket #
##############


etf_list = ['voo', 'vgt', 'vde', 'vpu', 'vdc', 'vfh', 'vht', 'vym', 'vox', 'vb', 'vo', 'vv', 'vug', 'vtv']


#########################################
# functions to calculate table features #
#########################################


# candle parts percentages
def candle_parts_pcts(o, c, h, l):
    full = h - l
    if full == 0:
        # If full is zero, return 0 for all components to avoid division by zero
        return 0, 0, 0
    body = abs(o - c)
    if o > c:
        top_wick = h - o
        bottom_wick = c - l
    else:
        top_wick = h - c
        bottom_wick = o - l
    return top_wick / full, body / full, bottom_wick / full


# calculates crossovers 
def crossover(slow_col, fast_col) -> int:
    if fast_col > slow_col:
        return 1
    else:
        return -1


# previous close and open gap % of pervious candle size
def gap_up_down_pct(o, pc, ph, pl):
    if (o == pc) or (ph == pl):
        return 0
    else:
        return (o - pc) / (ph - pl)
    
    
# z-score calculation
def zscore(x, mu, stdev):
    if stdev == 0:
        return 0
    else:
        return (x - mu) / stdev


# compute kelly criterion
def kelly_c(p, l=1, g=2.5):     
    return list(map(lambda x:(x / l - (1 - x) / g), p))

    
#############################################
# Functions for download and loading tables #
#############################################


def download(symbol: str, interval:str) -> None:

    try:
        stock = yf.Ticker(symbol)
        
        if interval in {'5m','15m','1h',}:
            interval_period_map = {'5m':58,
                                   '15m':58,
                                   '1h':728,
                                  }
            today = datetime.today().date()
            start = today - timedelta(days=interval_period_map[interval])
            stock_df = stock.history(interval=interval,
                                     start=str(start),
                                     end=None,
                                     # period=period,
                                     auto_adjust=False,
                                     prepost=True, # include aftermarket hours
                                    )
            
        else:
            stock_df = stock.history(interval=interval,
                             period='max',
                             auto_adjust=False,
                             prepost=True, # include aftermarket hours
                            )

        # Check if data is returned
        if stock_df.empty:
            logging.warning(f'No data found for ticker {symbol} with interval {interval}.')
            
        stock_df.columns = stock_df.columns.str.lower().str.replace(' ', '_')
        stock_df.to_pickle(f'./data_raw/{symbol}_{interval}_df.pkl')
        logging.info(f'Downloaded data for {symbol} successfully.')

        # Throttle requirests: wirte for 0.25 second before each call
        time.sleep(0.1)
        
    except Exception as e:
        logging.error(f'Failed to download ticker {symbol} due to: {e}')

    
def download_interval_process(interval: str, processes: int = 1) -> set():

    stocks_set = etf_top_stocks(*etf_list)

    params = [(symbol, interval) for symbol in stocks_set]

    with multiprocessing.Pool(processes=processes) as pool:
        pool.starmap(download, params)

    return stocks_set


def etf_top_stocks(*tickers: str) -> set:
    """
    Returns a set of top stock symbols from ETF tickers provided
    Parameters:
        tickers (str): One or more ETF ticker symbosl (e.g. 'xlk', 'vgt', 'spy').

    Returns:
        set: A distinct set of stock symbols from the top holdings of each ETF

    Sample Code:
    spy = yf.Ticker('XLK')
    spy_holdings_df = spy.get_funds_data().top_holdings
    set(spy_holdings_df.index)
    """
    top_stocks = set()

    for ticker in tickers:
        # convert ticker to uppercase
        ticker = ticker.upper()
        etf = yf.Ticker(ticker)

        try:
            # Retrieve the ETF funds data
            funds_data = etf.get_funds_data()
            if funds_data is not None and funds_data.top_holdings is not None:
                # Update the top_stocks set with the index values (stock symbols) from the top holdings Dataframe information retrieved
                top_stocks.update(set(funds_data.top_holdings.index))
        except Exception as e:
            print(f'Error fetching data for {ticker}: {e}')

    return top_stocks
    

def load_raw(symbol, interval):
    return pd.read_pickle(f'./data_raw/{symbol}_{interval}_df.pkl')


def load_model_df(symbol, interval):
    return pd.read_pickle(f'./data_transformed/{symbol}_{interval}_model_df.pkl')


#########################################
# functions to transform table features #
#########################################


def agg_interval_table(symbol: str, interval: str) -> None:
    ...


def make_table_features(symbol: str, interval: str) -> None:

    stock_table = load_raw(symbol, interval)

    # adj_close moving avgs sma 20, 10, 5
    stock_table['sma20'] = stock_table['adj_close'].rolling(window=20).mean().copy()
    stock_table['sma10'] = stock_table['adj_close'].rolling(window=10).mean().copy()
    stock_table['sma5'] = stock_table['adj_close'].rolling(window=5).mean().copy()
    stock_table['slow_sma_signal'] = stock_table.apply(lambda row: crossover(row['sma_20'], row['sma_10']), axis=1).copy()
    stock_table['fast_sma_signal'] = stock_table.apply(lambda row: crossover(row['sma_10'], row['sma_5']), axis=1).copy()

    # adj_close moving stdev 20, 10, 5
    stock_table['stdev20'] = stock_table['adj_close'].rolling(window=20).std().copy()
    stock_table['stdev10'] = stock_table['adj_close'].rolling(window=10).std().copy()
    stock_table['stdev5'] = stock_table['adj_close'].rolling(window=5).std().copy()

    # volume moving stdev 20, 10, 5
    stock_table['vol_stdev20'] = stock_table['volume'].rolling(window=20).std().copy()
    stock_table['vol_stdev10'] = stock_table['volume'].rolling(window=10).std().copy()
    stock_table['vol_stdev5'] = stock_table['volume'].rolling(window=5).std().copy()




if __name__ == '__main__':
    ...