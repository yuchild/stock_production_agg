#!/usr/bin/env python3
import streamlit as st
st.set_page_config(layout="wide")
from src import modules as f
import os
import pandas as pd
# import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import joblib
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def main():
    # App title
    st.title("Stock Next Interval Prediction App by [David Yu](https://www.linkedin.com/in/chi-yu)")

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
    account_balance = st.sidebar.text_input("Account Balance:", value='100000')
    max_loss_pct = st.sidebar.text_input("Max Loss % per trade (default 1.5%):", value='0.015')
    st.sidebar.markdown("""
    #### Validation Metrics
    - Percent Correct: Correctly predicted next interval in each of the last 100 entries, > 33.33% means model is better than random guessing
    - Kelly Criterion: Maximum fraction of account balance to risk per trade 2:1 ratio of win to loss
    #### Trade Metrics
    - Risk Amount: Amount to risk per trade based on Kelly Criterion and account balance
    - Last Price: Last price of the stock
    - Max Loss Amount: Maximum loss amount based on account balance and max loss percentage
    - Shares to Buy: Number of shares to buy based on risk amount and last price
    - Stop Loss Price: Price to set for stop loss based on entry (last) price and max loss %
    - Take Profit Price: Price to set for take profit based on last price and 2x max loss % above entry price for 2:1 win/loss ratio
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

    # after you have `probs` and before you compute stop_loss_price/take_profit_price
    # 1) rank the probabilities, highest first
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    # 2) if highest is no_change (index 0), pick the next best
    chosen_idx = ranked[1] if ranked[0] == 0 else ranked[0]
    # map to “up” or “down”
    direction = {1: "up", 2: "down"}[chosen_idx]

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

    if shares > 0:
        if direction == "up":
            stop_loss_price   = last_price - max_loss_amount / shares
            take_profit_price = last_price + 2 * max_loss_amount / shares
        else:  # down
            stop_loss_price   = last_price + max_loss_amount / shares
            take_profit_price = last_price - 2 * max_loss_amount / shares
    else:
        stop_loss_price = take_profit_price = 0
    # stop_loss_price = last_price - max_loss_amount / shares if shares > 0 else 0
    # take_profit_price = last_price + 2 * max_loss_amount / shares if shares > 0 else 0

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

            # decide label for shares
            action = "Buy" if direction == "up" else "Sell"
            color  = "green" if action == "Buy" else "red"

            # big, bold “Shares to Buy/Sell: N” with dynamic color
            st.markdown(
                f"<div style='font-size:32px; font-weight:600;'>"
                f"Shares to <span style='color:{color};'>{action}</span>: {shares}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # big, bold Stop Loss Price in red
            st.markdown(
                f"<div style='font-size:32px; font-weight:600;'>"
                f"Stop Loss Price: <span style='color:red;'>${stop_loss_price:.2f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # big, bold Take Profit Price in green
            st.markdown(
                f"<div style='font-size:32px; font-weight:600;'>"
                f"Take Profit Price: <span style='color:green;'>${take_profit_price:.2f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # st.metric("Stop Loss Price", f"${stop_loss_price:.2f}")
            # st.metric("Take Profit Price", f"${take_profit_price:.2f}")

        # Summary & Prospect
        with text_col2:
            st.header("📈 Summary")
            st.write(f"Symbol: {symbol}")
            st.write(f"Interval: {interval}")
            st.metric(f"Last Price: ", f"${last_price:.2f}")
            # st.header("Model Inferences")
            # st.subheader(f"Predicted Next {interval} Movement: ***{pred_label}***")
            # st.header("Model Probabilities:")
            # st.subheader(f"➡️ no_change: {probs[0]:.4f}")
            # st.subheader(f"⬆️ up: {probs[1]:.4f}")
            # st.subheader(f"⬇️ down: {probs[2]:.4f}")
            st.header("Model Inferences")
            # Color‑coded prediction label
            if pred_label.lower() == "up":
                label_color = "green"
            elif pred_label.lower() == "down":
                label_color = "red"
            else:
                label_color = "yellow"
            st.markdown(
                f"<h3>Predicted Next {interval} Direction: <span style='color:{label_color}'><b>{pred_label}</b></span></h3>",
                unsafe_allow_html=True,
            )
            st.header("Model Probabilities:")
            # Color‑coded probabilities
            st.markdown(
                f"⬆️ up: <span style='color:green'><b>{probs[1]:.4f}</b></span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"➡️ no_change: <span style='color:yellow'><b>{probs[0]:.4f}</b></span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"⬇️ down: <span style='color:red'><b>{probs[2]:.4f}</b></span>",
                unsafe_allow_html=True,
            )
            st.metric(f"Last Entry (EST): ", f"{dt_est.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")
            st.metric(f"Last Entry (PDT): ", f"{dt_pdt.strftime('%Y-%m-%d %I:%M:%S %p %Z')}")

    # Right: single, larger chart
    with col_right:
        st.header(f"{symbol} {interval} Adj Close + 15 Intervals Forecast")

        # 1) Call the existing function to build the figure
        f.plot_adj_close(symbol, interval)

        # 2) Grab the current Matplotlib figure and force a black background
        fig = plt.gcf()
        # Figure background
        fig.patch.set_facecolor('black')
        # Loop over all axes and set their backgrounds/labels to white
        for ax in fig.axes:
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            # If there is a legend, force its text/frame to white/black
            leg = ax.get_legend()
            if leg:
                for txt in leg.get_texts():
                    txt.set_color('white')
                leg.get_frame().set_facecolor('black')
                leg.get_frame().set_edgecolor('white')

        # 3) Render in Streamlit
        st.pyplot(fig)

if __name__ == '__main__':
    main()