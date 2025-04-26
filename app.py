#!/usr/bin/env python3

import streamlit as st
st.set_page_config(layout="wide")
from src import modules as f
import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def main():
    # App title
    st.title("Stock Analysis App")

    # Ensure directories exist for data and models
    os.makedirs('./data_raw', exist_ok=True)
    os.makedirs('./data_transformed', exist_ok=True)
    os.makedirs('./models', exist_ok=True)

    # Sidebar inputs
    st.sidebar.header("Input Parameters")
    symbol = st.sidebar.text_input("Stock Symbol", value="NVDA").upper()
    interval = st.sidebar.selectbox(
        "Interval", 
        options=['5m', '15m', '1h', '1d', '1wk', '1m',], 
        index=3
    )

    # Determine start date based on interval
    today = datetime.today().date()
    if interval in ['5m', '15m', '1h']:
        start_date = today - timedelta(days=14)
    elif interval == '1d':
        start_date = today - timedelta(days=90)
    else:
        start_date = today - timedelta(days=730)

    # Download raw data if missing
    raw_path = f'./data_raw/{symbol}_{interval}_df.pkl'
    if not os.path.exists(raw_path):
        f.download(symbol, interval)
    df_raw = f.load_raw(symbol, interval)
    last_price = df_raw['close'].iloc[-1]

    # Generate feature table if missing
    feat_path = f'./data_transformed/{symbol}_{interval}_model_df.pkl'
    if not os.path.exists(feat_path):
        # Ensure VIX features exist
        f.download('^VIX', interval)
        f.make_table_features(symbol, interval, build=False)
    df_feat = f.load_model_df(symbol, interval)

    # Prepare features for prediction
    feature_cols = [
        'slow_sma_signal','fast_sma_signal','stdev20','stdev10','stdev5',
        'vix_stdev20','vix_stdev10','vix_stdev5','vol_stdev20','vol_stdev10',
        'vol_stdev5','top_stdev20','top_stdev10','top_stdev5','body_stdev20',
        'body_stdev10','body_stdev5','bottom_stdev20','bottom_stdev10',
        'bottom_stdev5','pct_gap_up_down_stdev20','pct_gap_up_down_stdev10',
        'pct_gap_up_down_stdev5','month_of_year','day_of_month','day_of_week',
        'hour_of_day','candle_cluster','direction'
    ]
    df_feat = df_feat[feature_cols]
    X_input = df_feat.drop(columns=['direction']).iloc[[-1]]

    # Load model or notify if missing
    model_path = f'./models/xgboost_{interval}_model.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model not found for interval {interval}. Please train and upload the model file.")
        return
    model = joblib.load(model_path)

    # Make prediction
    pred = model.predict(X_input)[0]
    probs = model.predict_proba(X_input)[0]
    label_mapping = {0: "no_change", 2: "down", 1: "up"}
    pred_label = label_mapping.get(pred, "unknown").upper()

    # Datetime conversion
    dt_utc = X_input.index[0]
    dt_est = dt_utc.astimezone(ZoneInfo("America/New_York"))
    dt_pdt = dt_utc.astimezone(ZoneInfo("America/Los_Angeles"))

    # Compute validation metrics
    df_val = f.model_validation(symbol, interval)
    correctly = int(df_val.dir_pred_match.sum())
    entries = int(df_val.shape[0])
    p = correctly / entries if entries else 0
    kelly = f.kelly_c(p=[p], l=1, g=2)[0]

    # Split main view into two equal columns
    col_left, col_right = st.columns(2)

    # Left: text info with sub-columns for metrics and summary
    with col_left:
        text_col1, text_col2 = st.columns(2)
        # Metrics
        with text_col1:
            st.header("Validation Metrics")
            st.metric("Correctly Predicted", correctly)
            st.metric("Entries Predicted", entries)
            st.metric("Percent Correct", f"{p:.2%}")
            st.metric("Kelly Criterion (Risk This % for 2 to 1 win loss ratio)", f"{kelly:.2%}")
        # Summary & Prospect
        with text_col2:
            st.header("Summary")
            st.markdown(f"**Symbol:** {symbol}")
            st.markdown(f"**Interval:** {interval}")
            st.markdown(f"**Last Price:** {last_price:.2f}")
            st.header("Prospect Results")
            st.markdown(f"**Predicted Next {interval} Movement:** *{pred_label}*")
            st.markdown("**Prediction Probabilities:**")
            st.markdown(f"➡️ no_change: {probs[0]:.4f}")
            st.markdown(f"⬆️ up: {probs[1]:.4f}")
            st.markdown(f"⬇️ down: {probs[2]:.4f}")
            st.markdown(f"**Last Entry (EST):** {dt_est.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
            st.markdown(f"**Last Entry (PDT):** {dt_pdt.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")

    # Right: single, larger chart
    with col_right:
        st.header(f"{symbol} Price Chart")
        df_hist = yf.Ticker(symbol).history(
            interval=interval,
            start=str(start_date),
            end=str(today),
            auto_adjust=False,
            prepost=True
        )
        df_hist.columns = df_hist.columns.str.lower()
        df_hist.drop(['dividends', 'stock_splits'], axis=1, errors='ignore', inplace=True)
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        ax.plot(df_hist.index, df_hist['close'], linewidth=3)
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Price', color='white')
        ax.set_title(f"{symbol} Close Price History", color='white')
        ax.tick_params(colors='white')
        fig.autofmt_xdate()
        st.pyplot(fig)

if __name__ == '__main__':
    main()