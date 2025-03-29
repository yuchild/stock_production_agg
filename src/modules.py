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
import os
import pickle
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import classification_report

from datetime import datetime
from zoneinfo import ZoneInfo

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
        return 0


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
        stock_df = stock_df.drop(['dividends', 'stock_splits'], axis=1)
        stock_df.to_pickle(f'./data_raw/{symbol}_{interval}_df.pkl')
        logging.info(f'Downloaded data for {symbol} successfully.')
    
    except Exception as e:
        logging.error(f'Failed to download ticker {symbol} due to: {e}')

    
def download_interval_process(interval: str, processes: int = 1) -> set():

    stocks_set = etf_top_stocks(*etf_list)
    stocks_set.update({'^VIX'})

    params = [(symbol, interval) for symbol in stocks_set]

    with multiprocessing.Pool(processes=processes) as pool:
        pool.starmap(download, params)

    return stocks_set
    

def etf_top_stocks(*tickers: str) -> set():
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

def make_master_table(stock_list: set, interval: str = '1d') -> None:
    data_path = "./data_transformed"
    dfs = []

    for ticker in stock_list:
        file_name = f"{ticker}_{interval}_model_df.pkl"
        file_path = os.path.join(data_path, file_name)

        # Load the pickle file into a DataFrame
        df = pd.read_pickle(file_path)

        # Drop the last row
        df = df.iloc[:-1]

        # Collect the adjusted DataFrame
        dfs.append(df)

    # Combine all dataframes into one
    master_df = pd.concat(dfs, ignore_index=True)

    # Save the combined DataFrame to a new pickle file
    out_file_name = f"all_{interval}_model_df.pkl"
    out_file_path = os.path.join(data_path, out_file_name)
    master_df.to_pickle(out_file_path)

    print(f"Saved combined dataframe to: {out_file_path}")


def make_table_features(symbol: str, interval: str) -> None:

    # load vix table
    vix_table = load_raw('^VIX', interval)

    # vix moving stdev 20, 10, 5
    vix_table['vix_stdev20'] = vix_table['adj_close'].rolling(window=20).std().copy()
    vix_table['vix_stdev10'] = vix_table['adj_close'].rolling(window=10).std().copy()
    vix_table['vix_stdev5'] = vix_table['adj_close'].rolling(window=5).std().copy()

    # load stock table
    stock_table = load_raw(symbol, interval)
    
    # adj_close moving avgs sma 20, 10, 5
    stock_table['sma20'] = stock_table['adj_close'].rolling(window=20).mean().copy()
    stock_table['sma10'] = stock_table['adj_close'].rolling(window=10).mean().copy()
    stock_table['sma5'] = stock_table['adj_close'].rolling(window=5).mean().copy()
    stock_table['slow_sma_signal'] = stock_table.apply(lambda row: crossover(row['sma20'], row['sma10']), axis=1).copy()
    stock_table['fast_sma_signal'] = stock_table.apply(lambda row: crossover(row['sma10'], row['sma5']), axis=1).copy()

    # adj_close moving stdev 20, 10, 5
    stock_table['stdev20'] = stock_table['adj_close'].rolling(window=20).std().copy()
    stock_table['stdev10'] = stock_table['adj_close'].rolling(window=10).std().copy()
    stock_table['stdev5'] = stock_table['adj_close'].rolling(window=5).std().copy()

    # volume moving stdev 20, 10, 5
    stock_table['vol_stdev20'] = stock_table['volume'].rolling(window=20).std().copy()
    stock_table['vol_stdev10'] = stock_table['volume'].rolling(window=10).std().copy()
    stock_table['vol_stdev5'] = stock_table['volume'].rolling(window=5).std().copy()

     # candle parts %'s
    stock_table[['pct_top_wick', 'pct_body', 'pct_bottom_wick']] = stock_table.apply(lambda row: candle_parts_pcts(row['open'], row['close'], row['high'],  row['low']), axis=1, result_type='expand').copy()

    # stdev of candel parts 20, 10, 5
    stock_table['top_stdev20'] = stock_table['pct_top_wick'].rolling(window=20).std().copy() 
    stock_table['body_stdev20'] = stock_table['pct_body'].rolling(window=20).std().copy() 
    stock_table['bottom_stdev20'] = stock_table['pct_bottom_wick'].rolling(window=20).std().copy()

    stock_table['top_stdev10'] = stock_table['pct_top_wick'].rolling(window=10).std().copy() 
    stock_table['body_stdev10'] = stock_table['pct_body'].rolling(window=10).std().copy() 
    stock_table['bottom_stdev10'] = stock_table['pct_bottom_wick'].rolling(window=10).std().copy()

    stock_table['top_stdev5'] = stock_table['pct_top_wick'].rolling(window=5).std().copy() 
    stock_table['body_stdev5'] = stock_table['pct_body'].rolling(window=5).std().copy() 
    stock_table['bottom_stdev5'] = stock_table['pct_bottom_wick'].rolling(window=5).std().copy()

    # % gap btwn current open relative to previous candle size
    stock_table['pc'] = stock_table['close'].shift(1).copy()
    stock_table['ph'] = stock_table['high'].shift(1).copy()
    stock_table['pl'] = stock_table['low'].shift(1).copy()
    stock_table['pct_gap_up_down'] = stock_table.apply(lambda row: gap_up_down_pct(row['open'], row['pc'], row['ph'], row['pl']), axis=1, result_type='expand').copy()

    stock_table['pct_gap_up_down_stdev20'] = stock_table['pct_gap_up_down'].rolling(window=20).std().copy()
    stock_table['pct_gap_up_down_stdev10'] = stock_table['pct_gap_up_down'].rolling(window=10).std().copy()
    stock_table['pct_gap_up_down_stdev5'] = stock_table['pct_gap_up_down'].rolling(window=5).std().copy()

    # day of month, week, hour of day
    stock_table['month_of_year'] = stock_table.index.month     # Month of year
    stock_table['day_of_month'] = stock_table.index.day        # Day of the month (1-31)
    stock_table['day_of_week'] = stock_table.index.weekday     # Day of the week (0 = Monday, 6 = Sunday)
    stock_table['hour_of_day'] = stock_table.index.hour        # Hour of the day (0-23)

    # target column: direction: -1, 0, 1
    stock_table['adj_close_pctc'] = stock_table['adj_close'].pct_change(fill_method=None)
    stock_table['direction'] = pd.qcut(stock_table['adj_close_pctc'], q=3, labels=[2, 0, 1]) #
    stock_table['direction'] = stock_table['direction'].shift(-1).copy() # shift up to predict next time interval 

    # merge vix with stock table
    df_merged = pd.merge(stock_table, 
                         vix_table[['vix_stdev20', 'vix_stdev10', 'vix_stdev5']], 
                         left_index=True, 
                         right_index=True, 
                         how='left'
                        )
    
    # clustering... select columns ending with 'z##'
    z_columns = ['pct_top_wick', 'pct_body', 'pct_bottom_wick', 'pct_gap_up_down']
    
    # drop nulls for kmeans fit
    data_z = df_merged[z_columns].dropna().copy() 

    # KMeans stratification
    optimal_k = 3  # Replace with the optimal number from the elbow plot
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    data_z['cluster'] = kmeans.fit_predict(data_z)
    
    # Add the 'cluster' column back to the original DataFrame
    df_merged['candle_cluster'] = data_z['cluster']
    
    # save table for model building
    cols = ['open', 
            'high', 
            'low', 
            'close', 
            'adj_close',
            'volume',
            'slow_sma_signal',
            'fast_sma_signal',
            'stdev20',
            'stdev10',
            'stdev5',
            'vix_stdev20', 
            'vix_stdev10', 
            'vix_stdev5',
            'vol_stdev20',
            'vol_stdev10',
            'vol_stdev5',
            'top_stdev20',
            'top_stdev10',
            'top_stdev5',
            'body_stdev20',
            'body_stdev10',
            'body_stdev5',
            'bottom_stdev20',
            'bottom_stdev10',
            'bottom_stdev5',
            'pct_gap_up_down_stdev20',
            'pct_gap_up_down_stdev10',
            'pct_gap_up_down_stdev5',
            'month_of_year',
            'day_of_month',
            'day_of_week',
            'hour_of_day',
            'candle_cluster',
            'direction',
            ]
    df_merged[cols].iloc[:-50].to_pickle(f'./data_transformed/{symbol}_{interval}_model_df.pkl')


def make_table_features_process(stock_set: set(), interval: str, processes: int = 1) -> None:

    # stocks_set = etf_top_stocks(*etf_list)

    params = [(symbol, interval) for symbol in stock_set]

    with multiprocessing.Pool(processes=processes) as pool:
        pool.starmap(make_table_features, params)

    
##########################################
# functions to model transformed dataset #
##########################################


def xg_boost_model(interval="1d") -> None:
    # Path to the combined DataFrame
    combined_df_path = f"./data_transformed/all_{interval}_model_df.pkl"
    # Path to save the trained model
    model_path = f"./models/xgboost_{interval}_model.pkl"
    
    # 1. Load Data
    df = pd.read_pickle(combined_df_path)

    # 2. Define columns (in case there are extra columns)
    cols = ['slow_sma_signal',
            'fast_sma_signal',
            'stdev20',
            'stdev10',
            'stdev5',
            'vix_stdev20', 
            'vix_stdev10', 
            'vix_stdev5',
            'vol_stdev20',
            'vol_stdev10',
            'vol_stdev5',
            'top_stdev20',
            'top_stdev10',
            'top_stdev5',
            'body_stdev20',
            'body_stdev10',
            'body_stdev5',
            'bottom_stdev20',
            'bottom_stdev10',
            'bottom_stdev5',
            'pct_gap_up_down_stdev20',
            'pct_gap_up_down_stdev10',
            'pct_gap_up_down_stdev5',
            'month_of_year',
            'day_of_month',
            'day_of_week',
            'hour_of_day',
            'candle_cluster',
            'direction',
            ]
    df = df[cols]

    # 3. Split into features and target
    X = df.drop(columns=['direction'])
    y = df['direction']

    # 4. Define categorical columns and their transformer.
    categorical_cols = [
        'slow_sma_signal',
        'fast_sma_signal',
        'month_of_year',
        'day_of_month',
        'day_of_week',
        'hour_of_day',
        'candle_cluster',
    ]
    # Here we set sparse_output to False so that subsequent feature selection works on a dense array.
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )

    preprocessor = ColumnTransformer([
        ("cat", categorical_transformer, categorical_cols),
    ])

    # 5. Set up the feature selection step.
    # We use an XGBoost estimator with L1 regularization (reg_alpha > 0) so that less important features
    # are given coefficients of (or near) zero.
    xgb_selector = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=8,
        reg_alpha=1.0,  # L1 regularization parameter
        reg_lambda=1.0, # L2 regularization parameter (optional)
    )
    # The threshold parameter ('median') means features with importance below the median will be discarded.
    feature_selector = SelectFromModel(estimator=xgb_selector, threshold="median", prefit=False)

    # 6. Define the main classifier.
    xgb_clf = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
        n_jobs=8,
        reg_alpha=1.0,
        reg_lambda=1.0,
    )

    # 7. Build the pipeline.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", feature_selector),
        ("classifier", xgb_clf),
    ])

    # 8. Train/test split & fit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)

    # 9. Evaluate the model.
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 10. Get feature importances from the classifier.
    # Get the original feature names from the preprocessor.
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    # Retrieve the support mask from the feature selection step.
    mask = pipeline.named_steps["feature_selection"].get_support()
    selected_features = feature_names[mask]
    importances = pipeline.named_steps["classifier"].feature_importances_

    fi_df = pd.DataFrame({
        "feature": selected_features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\nFeature Importances (Descending):")
    print(fi_df)

    # 11. Save the trained pipeline.
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved at: {model_path}")
    

##########################################
# functions to use model for predictions #
##########################################

def model_prospect(symbol: str, interval: str) -> None:

    # Assume these functions are defined elsewhere
    download(symbol, interval)
    make_table_features(symbol, interval)
    df_prospect = load_model_df(symbol, interval)
    
    # Define columns (make sure these match your data)
    cols = ['slow_sma_signal',
            'fast_sma_signal',
            'stdev20',
            'stdev10',
            'stdev5',
            'vix_stdev20', 
            'vix_stdev10', 
            'vix_stdev5',
            'vol_stdev20',
            'vol_stdev10',
            'vol_stdev5',
            'top_stdev20',
            'top_stdev10',
            'top_stdev5',
            'body_stdev20',
            'body_stdev10',
            'body_stdev5',
            'bottom_stdev20',
            'bottom_stdev10',
            'bottom_stdev5',
            'pct_gap_up_down_stdev20',
            'pct_gap_up_down_stdev10',
            'pct_gap_up_down_stdev5',
            'month_of_year',
            'day_of_month',
            'day_of_week',
            'hour_of_day',
            'candle_cluster',
            'direction']
    
    df_prospect = df_prospect[cols].copy()  # Use only the needed columns
    
    # Prepare the raw feature input (drop the target column)
    X = df_prospect.drop(columns=['direction'])
    
    # Instead of converting to a NumPy array, preserve DataFrame structure
    X_input = X.iloc[[-1]]
    
    # Load the saved pipeline model.
    model = joblib.load(f'./models/xgboost_{interval}_model.pkl')
    
    # Use the saved pipeline to predict (it will perform its own transformation).
    prediction = model.predict(X_input)
    probabilities = model.predict_proba(X_input)
    
    # Map numerical predictions to text labels.
    label_mapping = {0: "no_change", 2: "down", 1: "up"} # 
    predicted_label = label_mapping.get(prediction[0], "unknown")
    
    print(f"\nModel Prediction {interval} {symbol.upper()}:")
    print(f"Predicted Next {interval} Movement: {predicted_label.upper()}")
    
    print("\nModel Prediction Probabilities:")
    for class_num, prob in zip([0, 1, 2], probabilities[0]): #
        print(f"{label_mapping[class_num]} ({class_num}): {prob:.4f}")

    print(f"\nLast Entry {interval} {symbol.upper()} Datetime Used for Prediction:\nNOTE: It's in or contains the full {interval} time interval.")
    # Original UTC time
    dt_utc = datetime.fromisoformat(str(X_input.index[0]))
    # Convert to Eastern Time (ET) and Pacific Time (PT)
    dt_est = dt_utc.astimezone(ZoneInfo("America/New_York"))
    dt_pdt = dt_utc.astimezone(ZoneInfo("America/Los_Angeles"))
    # Format with AM/PM
    print("EST:", dt_est.strftime('%Y-%m-%d %I:%M:%S %p %Z%z'))
    print("PDT:", dt_pdt.strftime('%Y-%m-%d %I:%M:%S %p %Z%z'))


def model_validation(symbol: str, interval: str):
    
    # Assume these functions are defined elsewhere
    download(symbol, interval)
    make_table_features(symbol, interval)
    df_prospect = load_model_df(symbol, interval)
    
    # Define columns (make sure these match your data)
    cols = ['slow_sma_signal',
            'fast_sma_signal',
            'stdev20',
            'stdev10',
            'stdev5',
            'vix_stdev20', 
            'vix_stdev10', 
            'vix_stdev5',
            'vol_stdev20',
            'vol_stdev10',
            'vol_stdev5',
            'top_stdev20',
            'top_stdev10',
            'top_stdev5',
            'body_stdev20',
            'body_stdev10',
            'body_stdev5',
            'bottom_stdev20',
            'bottom_stdev10',
            'bottom_stdev5',
            'pct_gap_up_down_stdev20',
            'pct_gap_up_down_stdev10',
            'pct_gap_up_down_stdev5',
            'month_of_year',
            'day_of_month',
            'day_of_week',
            'hour_of_day',
            'candle_cluster',
            'direction']
    
    df_prospect = df_prospect[cols].copy()  # Use only the needed columns
    
    # Prepare the raw feature input (drop the target column) for the last 50 rows (from -51 to -2)
    X_dir = df_prospect['direction'].iloc[-51:-1].copy()
    X = df_prospect.drop(columns=['direction'])
    X_input = X.iloc[-51:-1]  # selects 50 rows, excluding the very last row
    
    # Load the saved pipeline model.
    model = joblib.load(f'./models/xgboost_{interval}_model.pkl')
    
    # Predict using the pipeline on multiple rows.
    predictions = model.predict(X_input)
    probabilities = model.predict_proba(X_input)
    
    # Map numerical predictions to text labels.
    label_mapping = {0: "no_change", 2: "down", 1: "up"}
    X_dir_labels = [label_mapping.get(x, "unknown") for x in X_dir]
    predicted_labels = [label_mapping.get(pred, "unknown") for pred in predictions]
    
    # Create a copy of the corresponding rows from df_prospect
    df_validation = df_prospect.iloc[-51:-1].copy()
    
    # Add new columns for prediction and class probabilities.
    df_validation['direction'] = X_dir_labels
    df_validation['prediction'] = predicted_labels
    df_validation['no_change_prob'] = probabilities[:, 0]
    df_validation['up_prob'] = probabilities[:, 1]
    df_validation['down_prob'] = probabilities[:, 2]

    # Define the matching function
    def match_columns(val1, val2):
        return 1 if val1 == val2 else 0
        
    df_validation['dir_pred_match'] = df_validation.apply(lambda row: match_columns(row['direction'], row['prediction']), axis=1)
    
    return df_validation
    

def arima_model(symbol: str, interval: str) -> None:
    ...




if __name__ == '__main__':
    ...