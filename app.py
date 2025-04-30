#!/usr/bin/env python3
import streamlit as st
st.set_page_config(layout="wide")
from src import modules as f
import os
import pandas as pd
# import yfinance as yf
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
        options=['1m', '5m', '15m', '1h', '1d', '1wk', '1mo',], 
        index=3
    )
    account_balance = st.sidebar.text_input("Account Balance:", value='10000')
    max_loss_pct = st.sidebar.text_input("Max Loss % per trade (default 1%):", value='0.01')
    st.sidebar.markdown("""
    #### Validation Metrics
    - Percent Correct: Percentage of correctly predicted entries of last 100 entries
    - Kelly Criterion: Maximum fraction of account balance to risk per trade 2:1 ratio of win to loss
    #### Trade Metrics
    - Risk Amount: Amount to risk per trade based on Kelly Criterion and account balance
    - Last Price: Last price of the stock
    - Max Loss Amount: Maximum loss amount based on account balance and max loss percentage
    - Shares to Buy: Number of shares to buy based on risk amount and last price
    - Stop Loss Price: Price to set for stop loss based on last price and max loss amount
    - Take Profit Price: Price to set for take profit based on last price and max loss amount
    """)

    # Determine start date based on interval
    today = datetime.today().date()
    if interval in ['5m', '15m', '1h']:
        start_date = today - timedelta(days=13)
    elif interval == '1m':
        start_date = today - timedelta(days=2)
    elif interval == '1d':
        start_date = today - timedelta(days=90)
    else:
        start_date = today - timedelta(days=730)

    # Download raw data 
    f.download(symbol, interval)
    f.download('^VIX', interval)
    f.make_table_features(symbol, interval, build=False)
    df_raw = f.load_raw(symbol, interval)
    df_prospect = f.load_model_df(symbol, interval)
    last_price = df_prospect['close'].iloc[-1].copy()

    # Prepare features for prediction
    feature_cols = [
        'slow_sma_signal','fast_sma_signal','stdev20','stdev10','stdev5',
        'vix_stdev20','vix_stdev10','vix_stdev5','vol_stdev20','vol_stdev10',
        'vol_stdev5','top_stdev20','top_stdev10','top_stdev5','body_stdev20',
        'body_stdev10','body_stdev5','bottom_stdev20','bottom_stdev10',
        'bottom_stdev5','pct_gap_up_down_stdev20','pct_gap_up_down_stdev10',
        'pct_gap_up_down_stdev5','month_of_year','day_of_month','day_of_week',
        'hour_of_day','candle_cluster','direction',
    ]
    df_feat = df_prospect[feature_cols]
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
    risk = float(account_balance) * kelly
    max_loss_amount = risk * float(max_loss_pct)
    shares = int(risk / last_price) if risk > last_price else 0
    stop_loss_price = last_price - max_loss_amount / shares if shares > 0 else 0
    take_profit_price = last_price + 2 * max_loss_amount / shares if shares > 0 else 0

    # Split main view into two equal columns
    col_left, col_right = st.columns(2)

    # Left: text info with sub-columns for metrics and summary
    with col_left:
        text_col1, text_col2 = st.columns(2)
        # Metrics
        with text_col1:
            st.header("Validation Metrics")
            # st.metric("Correctly Predicted", correctly)
            # st.metric("Entries Predicted", entries)
            st.metric("Percent Correct", f"{p:.2%}")
            st.metric("Kelly Criterion", f"{kelly:.2%}")
            st.header("Trade Metrics")
            st.metric("Risk Amount", f"{risk:.2f}")
            st.metric("Last Price", f"${last_price:.2f}")
            st.metric("Max Loss Amount", f"{max_loss_amount:.2f}")
            st.metric("Shares to Buy", shares)
            st.metric("Stop Loss Price", f"${stop_loss_price:.2f}")
            st.metric("Take Profit Price", f"${take_profit_price:.2f}")
        # Summary & Prospect
        with text_col2:
            st.header("📈 Summary")
            st.write(f"Symbol: {symbol}")
            st.write(f"Interval: {interval}")
            st.metric(f"Last Price: ", f"${last_price:.2f}")
            st.header("Prospect Results")
            st.subheader(f"Predicted Next {interval} Movement: ***{pred_label}***")
            st.header("Prediction Probabilities:")
            st.subheader(f"➡️ no_change: {probs[0]:.4f}")
            st.subheader(f"⬆️ up: {probs[1]:.4f}")
            st.subheader(f"⬇️ down: {probs[2]:.4f}")
            st.metric(f"Last Entry (EST): ", f"{dt_est.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
            st.metric(f"Last Entry (PDT): ", f"{dt_pdt.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")

    # Right: single, larger chart
    with col_right:
        st.header(f"{symbol} {interval} Price Chart")
        # df_hist = yf.Ticker(symbol).history(
        #     interval=interval,
        #     start=str(start_date),
        #     end=str(today),
        #     auto_adjust=False,
        #     prepost=True
        # )

        # make start_date into a pandas Timestamp…
        start_ts = pd.Timestamp(start_date)
        # …and localize it if your index has a tz
        if df_prospect.index.tz is not None:
            start_ts = start_ts.tz_localize(df_prospect.index.tz)
        # now slice safely
        df_hist = df_prospect.loc[start_ts :].copy()
        df_hist.columns = df_hist.columns.str.lower()
        df_hist.drop(['dividends', 'stock_splits'], axis=1, errors='ignore', inplace=True)
        fig, ax = plt.subplots(figsize=(16, 16))
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