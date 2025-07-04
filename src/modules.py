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


# Optional: silence verbose GPU messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Enable memory growth on GPU to prevent allocation errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Using GPU: {gpus}")
    except RuntimeError as e:
        print(f"❌ GPU setup error: {e}")
else:
    print("⚠️ No GPU detected. Running on CPU.")

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score, ConfusionMatrixDisplay
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBClassifier, XGBRegressor


from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Set yfinance logging level to ERROR to suppress DEBUG logs
logging.getLogger("yfinance").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")


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
def kelly_c(p, l=1, g=2):     
    return list(map(lambda x:(x / l - (1 - x) / g), p))

    
#############################################
# Functions for download and loading tables #
#############################################


def download(symbol: str, interval:str) -> None:

    try:    
        stock = yf.Ticker(symbol)
        
        if interval in {'1m', '5m','15m','1h',}:
            interval_period_map = {'1m':5,
                                   '5m':58,
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


def make_table_features(symbol: str, interval: str, build: bool=True) -> None:

    # load vix table, make sure '^VIX' is up to date
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
    # build=True then df_merged[cols].iloc[:-50]
    # build=False then df_meraged[cols].iloc[:-1]
    
    if build:
        row = -101
    else:
        row = df_merged.shape[0]
        
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
    df_merged[cols].iloc[:row].to_pickle(f'./data_transformed/{symbol}_{interval}_model_df.pkl')


def make_table_features_process(stock_set: set(), interval: str, processes: int = 1, build: bool=True) -> None:

    # stocks_set = etf_top_stocks(*etf_list)

    params = [(symbol, interval, build==True) for symbol in stock_set]

    with multiprocessing.Pool(processes=processes) as pool:
        pool.starmap(make_table_features, params)

    
##########################################
# functions to model transformed dataset #
##########################################

def xg_boost_model(interval="1d", grid_search_on=False) -> None:
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
    # Set sparse_output to False so that subsequent feature selection works on a dense array.
    categorical_transformer = OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False
    )

    preprocessor = ColumnTransformer([
        ("cat", categorical_transformer, categorical_cols),
    ])

    # 5. Set up the feature selection step.
    # Use an XGBoost estimator with L1 regularization so that less important features are shrunk.
    xgb_selector = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=6,
        reg_alpha=1.0,
        reg_lambda=1.0,
    )
    feature_selector = SelectFromModel(estimator=xgb_selector, threshold="median", prefit=False)

    # 6. Define the main classifier.
    xgb_clf = XGBClassifier(
        objective="multi:softmax",
        num_class=3,
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=6,
        reg_alpha=1.0,
        reg_lambda=1.0,
        tree_method='hist',
    )

    # 7. Build the pipeline.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", feature_selector),
        ("classifier", xgb_clf),
    ])

    # 8. Train/test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 8a. Compute sample weights to tweak sensitivity.
    # This computes weights inversely proportional to class frequencies.
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    weight_dict = dict(zip(classes, class_weights))
    # Create a sample weight array for the training set.
    sample_weights = y_train.map(weight_dict)
    
    # 9. Optionally perform hyperparameter tuning via grid search.
    if grid_search_on:
        # Reduced parameter grid focusing on the most relevant hyperparameters.
        param_grid = {
            'feature_selection__threshold': ['median', 0.005],
            'classifier__max_depth': [3, 5],
            'classifier__learning_rate': [0.1, 0.01],
            'classifier__n_estimators': [100, 200],
            'classifier__reg_alpha': [0.0, 0.5],
            'classifier__reg_lambda': [1.0, 1.5]
        }
        # Using n_jobs=-1 to leverage all available cores for the grid search.
        grid_search = GridSearchCV(pipeline, 
                                   param_grid, 
                                   cv=3, 
                                   scoring='accuracy',
                                   n_jobs=6,
                                   verbose=3,
                                   pre_dispatch='2*n_jobs',
                                  )
        grid_search.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        print("Best parameters:", grid_search.best_params_)
        best_pipeline = grid_search.best_estimator_
    else:
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
        best_pipeline = pipeline

    # 10. Evaluate the best model.
    y_pred = best_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 11. Get feature importances from the classifier.
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    mask = best_pipeline.named_steps["feature_selection"].get_support()
    selected_features = feature_names[mask]
    importances = best_pipeline.named_steps["classifier"].feature_importances_

    fi_df = pd.DataFrame({
        "feature": selected_features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\nFeature Importances (Descending):")
    print(fi_df)

    # 12. Save the trained pipeline.
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_pipeline, model_path)
    print(f"\nModel saved at: {model_path}")

# neural net model
def neural_net_model(interval="1d"):
    # 1. Load data
    df_path = f"./data_transformed/all_{interval}_model_df.pkl"
    model_dir = f"./models"
    model_path = os.path.join(model_dir, f"neural_net_{interval}_model.keras")
    df = pd.read_pickle(df_path)

    # 2. Define features and target
    cols = [
        'slow_sma_signal','fast_sma_signal','stdev20','stdev10','stdev5',
        'vix_stdev20','vix_stdev10','vix_stdev5','vol_stdev20','vol_stdev10','vol_stdev5',
        'top_stdev20','top_stdev10','top_stdev5','body_stdev20','body_stdev10','body_stdev5',
        'bottom_stdev20','bottom_stdev10','bottom_stdev5',
        'pct_gap_up_down_stdev20','pct_gap_up_down_stdev10','pct_gap_up_down_stdev5',
        'month_of_year','day_of_month','day_of_week','hour_of_day',
        'candle_cluster','direction'
    ]
    df = df[cols].dropna()
    X = df.drop(columns=["direction"])
    y = df["direction"].astype(int)

    categorical_cols = [
        'slow_sma_signal','fast_sma_signal','month_of_year',
        'day_of_month','day_of_week','hour_of_day','candle_cluster'
    ]

    # 3. Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_cols)
    ], remainder="passthrough")

    X_processed = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, f"{model_dir}/neural_net_{interval}_preproc.pkl")

    # 4. Class weights for imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    weight_dict = dict(enumerate(class_weights))

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Convert y to categorical for Keras
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=3)

    # 7. Build model (transfer learning base)
    base_model = tf.keras.applications.DenseNet121(
        include_top=False, weights=None, input_shape=(X_train.shape[1], 1), pooling='avg'
    )

    # manually emulate transfer-learning-like behavior on tabular input
    model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Reshape((X_train.shape[1], 1)),
        layers.Conv1D(32, kernel_size=3, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(3, activation="softmax")
    ])

    # 8. Compile
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 9. Callbacks
    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(model_path, save_best_only=True)
    ]

    # 10. Train
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=50,
        batch_size=32,
        callbacks=cb,
        class_weight=weight_dict,
        verbose=2
    )

    # 11. Evaluation
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 12. Plot metrics
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['accuracy'], label='Train Acc')
    ax[0].plot(history.history['val_accuracy'], label='Val Acc')
    ax[0].legend(); ax[0].set_title('Accuracy')

    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Val Loss')
    ax[1].legend(); ax[1].set_title('Loss')
    plt.tight_layout()
    plt.savefig(f"{model_dir}/nn_{interval}_metrics.png")
    plt.close()

    print(f"\nSaved model to: {model_path}")
    print(f"Saved preprocessing pipeline to: {model_dir}/neural_net_{interval}_preproc.pkl")


# Train / save a 15-step-ahead regressor
def xg_boost_reg_model(interval: str = "1d", grid_search_on: bool = False) -> None:
    combined_df_path = f"./data_transformed/all_{interval}_model_df.pkl"
    model_path       = f"./models/xgboost_{interval}_regressor.pkl"
    horizon = 15

    # 1. load combined feature table
    df = pd.read_pickle(combined_df_path)

    # 2. build your multi-output target: next 15 adj_close values
    for i in range(1, horizon + 1):
        df[f"adj_close_t+{i}"] = df["adj_close"].shift(-i)
    df.dropna(subset=[f"adj_close_t+{horizon}"], inplace=True)

    # 3. define feature columns (same as your xg_boost_model, minus 'direction')
    feature_cols = [
        'open','high','low','close','adj_close','volume',
        'slow_sma_signal','fast_sma_signal',
        'stdev20','stdev10','stdev5',
        'vix_stdev20','vix_stdev10','vix_stdev5',
        'vol_stdev20','vol_stdev10','vol_stdev5',
        'top_stdev20','top_stdev10','top_stdev5',
        'body_stdev20','body_stdev10','body_stdev5',
        'bottom_stdev20','bottom_stdev10','bottom_stdev5',
        'pct_gap_up_down_stdev20','pct_gap_up_down_stdev10','pct_gap_up_down_stdev5',
        'month_of_year','day_of_month','day_of_week','hour_of_day',
        'candle_cluster'
    ]
    X = df[feature_cols]
    # y will be a DataFrame of shape (n_samples, 15)
    y = df[[f"adj_close_t+{i}" for i in range(1, horizon + 1)]]

    # 4. preprocessing: one‐hot encode the categorical signals, pass the rest through
    categorical_cols = [
        'slow_sma_signal','fast_sma_signal',
        'month_of_year','day_of_month','day_of_week','hour_of_day','candle_cluster'
    ]
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols)
    ], remainder="passthrough")

    # 5. set up the multi‐output XGB regressor
    base_reg = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=6,
        tree_method="hist"
    )
    multi_reg = MultiOutputRegressor(base_reg)

    # 6. build pipeline
    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("regressor", multi_reg)
    ])

    # 7. split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 8. optional grid search
    if grid_search_on:
        param_grid = {
            "regressor__estimator__n_estimators": [100, 200],
            "regressor__estimator__max_depth":    [3, 5],
            "regressor__estimator__learning_rate":[0.1, 0.01],
        }
        gs = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=6,
            verbose=2
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        print("Best params:", gs.best_params_)
    else:
        pipeline.fit(X_train, y_train)
        best = pipeline

    # 9. evaluate
    y_pred = best.predict(X_test)
    mse  = mean_squared_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}   R²: {r2:.4f}")

    # 10. save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best, model_path)
    print(f"Regressor saved to {model_path}")



##########################################
# functions to use model for predictions #
##########################################

def model_prospect(symbol: str, interval: str, build: bool=True) -> None:

    # Assume these functions are defined elsewhere
    download(symbol, interval)
    download('^VIX', interval)
    make_table_features(symbol, interval, build)
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
    X = df_prospect.drop(columns=['direction']).copy()
    
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
    
    # Load raw df to get the last row
    df_raw = load_raw(symbol, interval)

    print(f"\nModel Prediction {interval} {symbol.upper()}:")
    print(f"{symbol.upper()} last price: {df_raw['close'].iloc[-1]:.2f}")
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


def model_validation(symbol: str, interval: str, build: bool=True):
    
    # Assume these functions are defined elsewhere
    download(symbol, interval)
    download('^VIX', interval)
    make_table_features(symbol, interval, build)
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
    X_dir = df_prospect['direction'].iloc[-101:-1].copy()
    X = df_prospect.drop(columns=['direction'])
    X_input = X.iloc[-101:-1]  # selects 100 rows, excluding the very last row
    
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
    df_validation = df_prospect.iloc[-101:-1].copy()
    
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


#######################################
# functions for plots / visulizations #
#######################################

# xgboost regressor model plot the last 45 adj_close + next 15 predicted
def plot_adj_close(symbol: str,
                   interval: str = "1d",
                   past_intervals: int = 45,
                   future_intervals: int = 15):
    # 1. download + feature-build for this symbol
    download(symbol, interval)
    download("^VIX", interval)
    make_table_features(symbol, interval, build=False)
    df_feat = load_model_df(symbol, interval)

    # 2. historical series
    df_raw = load_raw(symbol, interval)
    ser_past = df_raw["adj_close"].tail(past_intervals)

    # 3. load regressor
    model_path = f"./models/xgboost_{interval}_regressor.pkl"
    reg = joblib.load(model_path)

    # 4. get the very last feature‐row
    X_input = df_feat.drop(columns=["direction"]).iloc[[-1]]

    # 5. predict multi‐step
    preds = reg.predict(X_input).flatten()

    # 6. build a future‐datetime index
    int_map = {
        "1d": timedelta(days=1),
        "1h": timedelta(hours=1),
        "15m": timedelta(minutes=15),
        "5m": timedelta(minutes=5),
        "1m": timedelta(minutes=1),
    }
    delta = int_map.get(interval, timedelta(days=1))
    last_ts = df_raw.index[-1]
    future_idx = [last_ts + delta * (i + 1) for i in range(future_intervals)]

    # 7. plot
    plt.figure(figsize=(10, 6))
    plt.plot(ser_past.index, ser_past.values, label="Historical adj_close")
    plt.plot(future_idx, preds,          label="Predicted adj_close", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Adj Close Price")
    plt.title(f"{symbol.upper()} — Last {past_intervals} + Next {future_intervals} {interval} Points")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    ...