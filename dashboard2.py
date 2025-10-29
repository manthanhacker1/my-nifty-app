# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import pandas_ta as ta
import time
from kiteconnect import KiteConnect
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go

# =========================================================================
# --- 💡 START: SAHI WALA SECRET CODE 💡 ---
# =========================================================================
# --- CONFIGURATION (UPDATED for Intraday Speed) ---
# Secrets ko unke NAAM se call karna hai, VALUE se nahi
# YEH SAHI HAI, PASTE KARO
API_KEY = st.secrets["API_KEY"]
API_SECRET = st.secrets["API_SECRET"]
ACCESS_TOKEN = st.secrets["ACCESS_TOKEN"]
REFRESH_INTERVAL_SECONDS = 15
# =========================================================================
# --- 💡 END: SAHI WALA SECRET CODE 💡 ---
# =========================================================================

# --- QUADRANT WEIGHTS (NOW DYNAMIC) ---
# Weights for a normal trading day
QUADRANT_WEIGHTS_NORMAL = {
    'price_volume': 0.30,  # Price vs VWAP, Open, Premium
    'derivatives': 0.45,   # PCR, OI Change, OI Skew
    'technicals': 0.25     # Supertrend, EMA, MACD, ADX
}
# Weights for Expiry Day (OI data is unreliable)
QUADRANT_WEIGHTS_EXPIRY = {
    'price_volume': 0.40,  # More weight on price action
    'derivatives': 0.25,   # Less weight on (unreliable) OI
    'technicals': 0.35     # More weight on the TA trend
}

# --- TECHNICALS CONFIG ---
TA_TIMEFRAME = "5minute"
TA_SUPERTREND_LEN = 10
TA_SUPERTREND_MULT = 3
TA_EMA_LEN = 50
VOLATILITY_PCT_THRESH = 0.2
TA_MACD_FAST = 12
TA_MACD_SLOW = 26
TA_MACD_SIGNAL = 9
TA_ADX_LEN = 14

# --- DERIVATIVES CONFIG ---
PCR_BULL_THRESH = 0.10
PCR_STRONG_BULL_THRESH = 0.20
PCR_BEAR_THRESH = -0.08
PCR_STRONG_BEAR_THRESH = -0.15

# --- LOGIC FIX: Increased OI Skew range to be more meaningful ---
OI_SKEW_RANGE = 250 # Was 50 (too small). 250 covers 5 strikes on either side.

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="🧠 Nifty Market Intelligence Matrix V2.1", layout="wide")
st.title("🧠 Nifty Market Intelligence Dashboard (Intraday)")
st.info("A comprehensive, real-time 3-Quadrant model for Nifty. Price/OI update every 15s, Technicals update every 5min.")

# --- KITE AUTH (CACHE REMOVED TO PREVENT AUTH FAILURE CACHING) ---
def authenticate():
    try:
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(ACCESS_TOKEN)
        kite.profile()
        return kite
    except Exception as e:
        error_msg = str(e)
        if "Incorrect api_key" in error_msg or "token" in error_msg or "session" in error_msg:
            st.error("🚨 Authentication Failed. Please ensure your ACCESS_TOKEN is **fresh** and all credentials are correct.", icon="🔥")
        else:
            st.error(f"Authentication Failed: {e}", icon="🚨")
        st.stop()

kite = authenticate()

# --- INSTRUMENT LOADING ---
@st.cache_data(ttl=timedelta(hours=1))
def load_instruments(_kite):
    try:
        nfo = pd.DataFrame(_kite.instruments('NFO'))
        nse = pd.DataFrame(_kite.instruments('NSE'))
        
        nifty_spot_token = nse[nse['tradingsymbol'] == 'NIFTY 50'].iloc[0]['instrument_token']

        nifty_opts = nfo[(nfo['name'] == 'NIFTY') & (nfo['segment'] == 'NFO-OPT')]
        valid_expiries = sorted([e for e in nifty_opts['expiry'].unique() if pd.to_datetime(e).date() >= datetime.now().date()])
        
        if not valid_expiries:
            st.error("No valid Nifty option expiries found.", icon="⚠️")
            return pd.DataFrame(), "", 0, None
            
        nearest_opt_expiry = valid_expiries[0]
        nifty_instruments = nifty_opts[nifty_opts['expiry'] == nearest_opt_expiry]

        nifty_fut = nfo[(nfo['name'] == 'NIFTY') & (nfo['segment'] == 'NFO-FUT')]
        valid_fut_expiries = sorted([e for e in nifty_fut['expiry'].unique() if pd.to_datetime(e).date() >= datetime.now().date()])
        if not valid_fut_expiries:
            st.error("No valid Nifty futures expiries found.", icon="⚠️")
            return pd.DataFrame(), "", 0, None

        nearest_fut_expiry = valid_fut_expiries[0]
        nifty_futures_symbol = nifty_fut[nifty_fut['expiry'] == nearest_fut_expiry].iloc[0]['tradingsymbol']

        return nifty_instruments, nifty_futures_symbol, nifty_spot_token, pd.to_datetime(nearest_opt_expiry)
    except Exception as e:
        st.error(f"FATAL: Could not fetch instruments: {e}", icon="📡")
        st.stop()

# --- MODIFIED: Get expiry date for dynamic weighting logic ---
nifty_instruments, nifty_futures_symbol, nifty_spot_token, nearest_opt_expiry = load_instruments(kite)
if nearest_opt_expiry is None:
    st.stop()

# =========================================================================
# --- 💡 START: API RATE LIMIT FIX 💡 ---
# This function is cached. It will only run ONCE every 5 minutes
# based on the 'time_bucket_key', saving thousands of API calls.
# =========================================================================
@st.cache_data(show_spinner=f"Updating {TA_TIMEFRAME} technicals...")
def get_technical_analysis(_kite, instrument_token, time_bucket_key):
    """
    Fetches and calculates technical indicators.
    This function is cached and only runs when 'time_bucket_key' changes.
    """
    results = {
        'Technical_Status': 'Error',
        'Supertrend_Value': 0, 'Supertrend_Signal': 0,
        'EMA_Value': 0, 'EMA_Signal': 0,
        'MACD_Histogram': 0, 'MACD_Signal': 0,
        'ADX_PDI_NPI_Signal': 0,
        'Technicals_Score': 0
    }
    
    try:
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=15)
        
        hist_data = _kite.historical_data(instrument_token, from_date, to_date, TA_TIMEFRAME)
        hist_df = pd.DataFrame(hist_data)
        hist_df['date'] = pd.to_datetime(hist_df['date'])
        
        if len(hist_df) < 3:
            raise Exception("Insufficient historical data (Need 3+ bars).")

        # Calculate Technicals
        hist_df.ta.supertrend(length=TA_SUPERTREND_LEN, multiplier=TA_SUPERTREND_MULT, append=True)
        hist_df.ta.ema(length=TA_EMA_LEN, append=True)
        hist_df.ta.macd(fast=TA_MACD_FAST, slow=TA_MACD_SLOW, signal=TA_MACD_SIGNAL, append=True)
        hist_df.ta.adx(length=TA_ADX_LEN, append=True)

        # CRITICAL: Use the last *CLOSED* candle for indicator values (iloc[-2])
        closed_row = hist_df.iloc[-2]
        
        # 1. Supertrend Signal (30%)
        st_signal_direction = closed_row.get(f'SUPERTd_{TA_SUPERTREND_LEN}_{TA_SUPERTREND_MULT}.0', 1)
        results['Supertrend_Value'] = closed_row.get(f'SUPERT_{TA_SUPERTREND_LEN}_{TA_SUPERTREND_MULT}.0', closed_row['close'])
        results['Supertrend_Signal'] = 1 if st_signal_direction == 1 else -1

        # 2. EMA Signal (25%)
        results['EMA_Value'] = closed_row.get(f'EMA_{TA_EMA_LEN}', closed_row['close'])
        closed_price = closed_row['close']
        results['EMA_Signal'] = 1 if closed_price > results['EMA_Value'] else -1

        # 3. MACD Signal (25%)
        macd_line = closed_row.get(f'MACD_{TA_MACD_FAST}_{TA_MACD_SLOW}_{TA_MACD_SIGNAL}', 0)
        signal_line = closed_row.get(f'MACDs_{TA_MACD_FAST}_{TA_MACD_SLOW}_{TA_MACD_SIGNAL}', 0)
        results['MACD_Histogram'] = closed_row.get(f'MACDh_{TA_MACD_FAST}_{TA_MACD_SLOW}_{TA_MACD_SIGNAL}', 0)
        prev_macd_hist = hist_df.iloc[-3].get(f'MACDh_{TA_MACD_FAST}_{TA_MACD_SLOW}_{TA_MACD_SIGNAL}', 0)
        
        if macd_line > signal_line and results['MACD_Histogram'] > prev_macd_hist:
            results['MACD_Signal'] = 1
        elif macd_line < signal_line and results['MACD_Histogram'] < prev_macd_hist:
            results['MACD_Signal'] = -1
        else:
            results['MACD_Signal'] = 0

        # 4. ADX/DI Signal (20%)
        dmp = closed_row.get(f'DMP_{TA_ADX_LEN}', 0)
        dmn = closed_row.get(f'DMN_{TA_ADX_LEN}', 0)
        results['ADX_PDI_NPI_Signal'] = 1 if dmp > dmn else -1

        # Calculate final Q3 score
        results['Technicals_Score'] = (
            results['Supertrend_Signal'] * 0.30 +
            results['EMA_Signal'] * 0.25 +
            results['MACD_Signal'] * 0.25 +
            results['ADX_PDI_NPI_Signal'] * 0.20
        )
        results['Technical_Status'] = "OK"

    except Exception as e:
        results['Technical_Status'] = f"Error: {str(e)}"
        
    return results
# =========================================================================
# --- 💡 END: API RATE LIMIT FIX 💡 ---
# =========================================================================


# --- UTILITY FUNCTIONS ---
def get_atm_strike(price, step=50):
    return round(price / step) * step

def score_factor_5pt(value, strong_bullish_thresh, bullish_thresh, strong_bearish_thresh, bearish_thresh):
    if value >= strong_bullish_thresh: return 2
    elif value >= bullish_thresh: return 1
    elif value <= strong_bearish_thresh: return -2
    elif value <= bearish_thresh: return -1
    return 0

def get_signal_narrative_5pt(score):
    if score >= 1.5: return "🚀 Strong Bullish"
    if score >= 0.5: return "📈 Bullish"
    if score > -0.5: return "⚖️ Neutral"
    if score > -1.5: return "📉 Bearish"
    return "🔻 Strong Bearish"

def get_signal_narrative_binary(score):
    if score == 1: return "🟢 Bullish (+1)"
    if score == -1: return "🔴 Bearish (-1)"
    return "🟡 Neutral (0)"

def get_final_narrative(score):
    if score >= 70: return "🚀 EXTREME BULLISH", "bullish"
    elif score >= 30: return "📈 BULLISH BIAS", "bullish"
    elif score > -30: return "⚖️ NEUTRAL/CHOPPY", "neutral"
    elif score > -70: return "📉 BEARISH BIAS", "bearish"
    else: return "🔻 EXTREME BEARISH", "bearish"

# --- CORE ANALYSIS (Modified to accept TA results and dynamic weights) ---
def analyze_market_matrix(kite, quotes, option_df, nifty_instruments, nifty_futures_symbol, ta_results, weights):
    matrix = {}
    detailed_metrics = {}

    option_data = pd.merge(
        option_df,
        nifty_instruments[['instrument_token', 'strike', 'instrument_type']],
        on='instrument_token'
    )

    nifty_spot = quotes.get("NSE:NIFTY 50", {})
    nifty_fut = quotes.get(f"NFO:{nifty_futures_symbol}", {})
    
    price = nifty_spot.get('last_price', 0)
    vwap = nifty_fut.get('average_price', 0)
    open_price = nifty_spot.get('ohlc', {}).get('open', price)
    
    if price == 0 or vwap == 0 or open_price == 0:
        return None, detailed_metrics

    # --- Q1: Price vs VWAP Logic (Dynamic Weight) - 5-POINT SCORING ---
    q1 = 0
    pv_vwap_pct = (price - vwap) / vwap * 100 if vwap > 0 else 0
    q1 += score_factor_5pt(pv_vwap_pct, VOLATILITY_PCT_THRESH * 2, VOLATILITY_PCT_THRESH, -VOLATILITY_PCT_THRESH * 2, -VOLATILITY_PCT_THRESH) * 0.50
    detailed_metrics['Price_VWAP_Pct'] = pv_vwap_pct
    
    pv_open_pct = (price - open_price) / open_price * 100 if open_price > 0 else 0
    q1 += score_factor_5pt(pv_open_pct, VOLATILITY_PCT_THRESH * 2, VOLATILITY_PCT_THRESH, -VOLATILITY_PCT_THRESH * 2, -VOLATILITY_PCT_THRESH) * 0.30
    detailed_metrics['Price_Open_Pct'] = pv_open_pct
    
    fut_prem = (nifty_fut.get('last_price', 0) - price)
    q1 += score_factor_5pt(fut_prem, 5, 3, -5, -3) * 0.20
    detailed_metrics['Futures_Premium'] = fut_prem
    
    matrix['price_volume'] = q1
    detailed_metrics['Price_Volume_Score'] = q1

    # --- Q2: Derivatives Analysis (Dynamic Weight) - 5-POINT SCORING ---
    df_ce = option_data[option_data['instrument_type'] == 'CE']
    df_pe = option_data[option_data['instrument_type'] == 'PE']

    df_ce['oi_change'] = df_ce['oi'] - df_ce['open_interest']
    df_pe['oi_change'] = df_pe['oi'] - df_pe['open_interest']
    
    q2 = 0
    
    # 1. Total PCR Score (40% Weight)
    total_ce_oi = df_ce['oi'].sum()
    total_pe_oi = df_pe['oi'].sum()
    pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1
    
    q2 += score_factor_5pt(pcr_oi - 1, PCR_STRONG_BULL_THRESH, PCR_BULL_THRESH, PCR_STRONG_BEAR_THRESH, PCR_BEAR_THRESH) * 0.40
    detailed_metrics['Total_PCR'] = pcr_oi
    
    # Filtered "Action Zone" Analysis (Around ATM)
    atm_strike = get_atm_strike(price)
    
    near_strikes_tokens = option_data[
        (option_data['strike'] >= atm_strike - OI_SKEW_RANGE) & 
        (option_data['strike'] <= atm_strike + OI_SKEW_RANGE)
    ]['instrument_token']

    near_strikes_ce = df_ce[df_ce['instrument_token'].isin(near_strikes_tokens)]
    near_strikes_pe = df_pe[df_pe['instrument_token'].isin(near_strikes_tokens)]

    # 2. Filtered OI Change Score (30% Weight)
    put_oi_change_near = near_strikes_pe['oi_change'].sum()
    call_oi_change_near = near_strikes_ce['oi_change'].sum()

    total_absolute_change = abs(put_oi_change_near) + abs(call_oi_change_near)
    
    if total_absolute_change > 0:
        net_oi_change_contribution = (put_oi_change_near - call_oi_change_near) / total_absolute_change
    else:
        net_oi_change_contribution = 0.0

    OI_CONTRIB_STRONG_THRESH = 0.5
    OI_CONTRIB_THRESH = 0.2

    q2 += score_factor_5pt(net_oi_change_contribution, 
                             OI_CONTRIB_STRONG_THRESH, 
                             OI_CONTRIB_THRESH, 
                             -OI_CONTRIB_STRONG_THRESH, 
                             -OI_CONTRIB_THRESH) * 0.30
    
    detailed_metrics['Near_OI_Net_Pct_Change'] = net_oi_change_contribution * 100

    # 3. ATM/Near OTM OI Skew Score (30% Weight)
    total_call_oi_near = near_strikes_ce['oi'].sum()
    total_put_oi_near = near_strikes_pe['oi'].sum()
    oi_skew_ratio = total_put_oi_near / total_call_oi_near if total_call_oi_near > 0 else 1.0

    q2 += score_factor_5pt(oi_skew_ratio - 1, 0.15, 0.05, -0.15, -0.05) * 0.30
    detailed_metrics['Near_OI_Skew_Ratio'] = oi_skew_ratio

    matrix['derivatives'] = q2
    detailed_metrics['Derivatives_Score'] = q2
    
    # --- Q3: Technical Analysis (Dynamic Weight) ---
    # Data is now pre-calculated and passed in as 'ta_results'
    detailed_metrics.update(ta_results) # Add all TA values to metrics
    matrix['technicals'] = ta_results.get('Technicals_Score', 0)

    # --- FINAL COMPOSITE SCORE (using DYNAMIC weights) ---
    final_score_raw = (
        matrix['price_volume'] * weights['price_volume'] +
        matrix['derivatives'] * weights['derivatives'] +
        matrix['technicals'] * weights['technicals']
    )
    
    # Scale from approx [-2, 2] to [-100, 100]
    final_score = int(np.clip((final_score_raw / 2) * 100, -100, 100))
    
    detailed_metrics['Live_Price'] = price
    detailed_metrics['VWAP'] = vwap
    detailed_metrics['Final_Score'] = final_score
    detailed_metrics['Timestamp'] = datetime.now()

    return matrix, detailed_metrics

# --- Helper function for plotting historical metrics ---
def create_metric_chart(chart_data_df, y_col, title, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data_df['Timestamp_Str'], 
        y=chart_data_df[y_col], 
        name=title, 
        mode='lines',
        line=dict(color=color, width=2)
    ))
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.5, font=dict(size=14)),
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title=None,
        yaxis_title=None,
        xaxis_showticklabels=False,
        hovermode="x unified"
    )
    return fig

# --- SESSION STATE MANAGEMENT ---
if 'score_history' not in st.session_state:
    st.session_state.score_history = pd.DataFrame(columns=[
        'Timestamp', 'Final_Score', 'Live_Price', 'Price_Volume_Score', 
        'Derivatives_Score', 'Technicals_Score', 'Supertrend_Signal', 'EMA_Signal', 'MACD_Signal',
        'Total_PCR', 'Near_OI_Net_Pct_Change', 'Near_OI_Skew_Ratio', 'Price_VWAP_Pct', 
        'Price_Open_Pct', 'Futures_Premium'
    ])

# --- DYNAMIC WEIGHTING LOGIC ---
is_expiry_day = (nearest_opt_expiry.date() == datetime.now().date())
current_weights = QUADRANT_WEIGHTS_EXPIRY if is_expiry_day else QUADRANT_WEIGHTS_NORMAL

if is_expiry_day:
    st.warning("🔥 **EXPIRY DAY DETECTED** - Weights adjusted. Reliance on OI data (Derivatives) is reduced.", icon="⚠️")

placeholder = st.empty()

# --- MAIN RUN LOOP ---
while True:
    try:
        # --- Data Fetching (Fast) ---
        symbols = ["NSE:NIFTY 50", f"NFO:{nifty_futures_symbol}"]
        quotes = kite.quote(symbols)

        tokens = nifty_instruments['instrument_token'].astype(str).tolist()
        option_quotes_raw = kite.quote(tokens)
        
        option_data_list = []
        for t, quote in option_quotes_raw.items():
            current_oi = quote.get('oi', 0)
            prev_eod_oi = quote.get('open_interest')
            if prev_eod_oi is None:
                prev_eod_oi = 0 # Default to 0 if API returns None
            
            option_data_list.append({
                'instrument_token': int(t.split(':')[-1]), 
                'oi': current_oi,
                'open_interest': prev_eod_oi
            })
        option_details_df = pd.DataFrame(option_data_list)
        
        # --- Technical Analysis (Smartly Cached) ---
        # 1. Create a "key" that only changes every 5 minutes.
        # We add a 15-second grace period to ensure the 5-min candle
        # data is available from the API before we try to fetch it.
        now = datetime.now()
        if now.second < 15 and now.minute % 5 == 0:
            # e.g., at 9:20:05, use the 9:15-9:19 bucket key
            bucket_minute = (now - timedelta(minutes=1)).minute
        else:
            # e.g., at 9:20:16, use the 9:20-9:24 bucket key
            bucket_minute = now.minute
        
        five_min_bucket_key = f"{now.date()}-{now.hour}-{bucket_minute // 5}"
        
        # 2. Call the cached function.
        # This will ONLY run the expensive API call and calculations
        # when 'five_min_bucket_key' changes (i.e., every 5 mins).
        ta_results = get_technical_analysis(kite, nifty_spot_token, five_min_bucket_key)
        
        # --- Core Analysis (Fast) ---
        matrix, detailed_metrics = analyze_market_matrix(
            kite, 
            quotes, 
            option_details_df, 
            nifty_instruments, 
            nifty_futures_symbol,
            ta_results,       # Pass in the cached TA results
            current_weights   # Pass in the dynamic weights
        )

        if matrix:
            
            # --- Update Session History ---
            new_row = pd.Series({
                'Timestamp': detailed_metrics['Timestamp'],
                'Final_Score': detailed_metrics['Final_Score'],
                'Live_Price': detailed_metrics['Live_Price'],
                'Price_Volume_Score': matrix['price_volume'],
                'Derivatives_Score': matrix['derivatives'],
                'Technicals_Score': matrix['technicals'],
                'Supertrend_Signal': detailed_metrics['Supertrend_Signal'],
                'EMA_Signal': detailed_metrics['EMA_Signal'],
                'MACD_Signal': detailed_metrics['MACD_Signal'],
                'Total_PCR': detailed_metrics['Total_PCR'],
                'Near_OI_Net_Pct_Change': detailed_metrics['Near_OI_Net_Pct_Change'],
                'Near_OI_Skew_Ratio': detailed_metrics['Near_OI_Skew_Ratio'],
                'Price_VWAP_Pct': detailed_metrics['Price_VWAP_Pct'],
                'Price_Open_Pct': detailed_metrics['Price_Open_Pct'],
                'Futures_Premium': detailed_metrics['Futures_Premium'],
            }).to_frame().T

            st.session_state.score_history = pd.concat([st.session_state.score_history, new_row], ignore_index=True)
            
            # Keep history for ~8.3 hrs
            if len(st.session_state.score_history) > 2000:
                st.session_state.score_history = st.session_state.score_history.iloc[-2000:]

            # --- RENDER UI ---
            with placeholder.container():
                
                # 1. HEADER AND MAIN SCORE
                score_display = detailed_metrics['Final_Score']
                narrative, bias = get_final_narrative(score_display)
                
                st.metric(f"Nifty Conviction Score ({nifty_futures_symbol.split('2')[0]})", f"{score_display}", narrative)
                
                st.markdown(f"""
                <style>
                .stProgress > div > div > div:nth-child(2) {{
                    background-color: {'#1D9F36' if bias == 'bullish' else '#FF4B4B' if bias == 'bearish' else '#F3CA10'};
                }}
                </style>
                """, unsafe_allow_html=True)
                
                progress_val = (score_display + 100) / 200
                st.progress(progress_val, text=f"**{narrative}**")
                st.divider()

                # 2. TABS
                tab1, tab2, tab3 = st.tabs(["📊 Dashboard & Score History", "⚙️ Technicals Deep Dive", "📋 Raw Data & Metrics"])
                
                chart_data = st.session_state.score_history.copy()
                chart_data['Timestamp'] = pd.to_datetime(chart_data['Timestamp'])
                chart_data['Timestamp_Str'] = chart_data['Timestamp'].dt.strftime('%H:%M:%S')

                # --- TAB 1: DASHBOARD ---
                with tab1:
                    st.header("Quadrant Summary")
                    cols = st.columns(3)
                    
                    cols[0].metric(
                        f"Price/Volume ({current_weights['price_volume']*100:.0f}%)", 
                        get_signal_narrative_5pt(matrix['price_volume'])
                    )
                    cols[1].metric(
                        f"Derivatives ({current_weights['derivatives']*100:.0f}%)", 
                        get_signal_narrative_5pt(matrix['derivatives']),
                        help="Note: Open Interest data from the exchange can be delayed by 3-5 minutes."
                    )
                    cols[2].metric(
                        f"Technicals ({current_weights['technicals']*100:.0f}%)", 
                        get_signal_narrative_5pt(matrix['technicals']),
                        help=f"Updates every 5 minutes based on the last closed {TA_TIMEFRAME} candle."
                    )

                    st.markdown("---")
                    st.subheader("Conviction Score History (Full Day)")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=chart_data['Timestamp_Str'], 
                        y=chart_data['Final_Score'], 
                        name='Conviction Score', 
                        mode='lines',
                        line=dict(color='#00C0F0', width=3),
                        yaxis='y1'
                    ))
                    fig.add_trace(go.Scatter(
                        x=chart_data['Timestamp_Str'], 
                        y=chart_data['Live_Price'], 
                        name='Nifty Live Price', 
                        mode='lines',
                        line=dict(color='#FF7043', dash='dot'),
                        yaxis='y2'
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        hovermode="x unified",
                        height=400,
                        margin=dict(l=20, r=20, t=30, b=20),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        xaxis_title="Time",
                        yaxis=dict(
                            title=dict(text='Conviction Score (-100 to +100)', font=dict(color='#BFBFBF')),
                            tickfont=dict(color='#BFBFBF'),
                            range=[-100, 100],
                            side='left'
                        ),
                        # =========================================================================
                        # --- 💡 START: SYNTAXERROR FIX (Already applied) 💡 ---
                        # =========================================================================
                        yaxis2=dict(
                            title=dict(text='Nifty Live Price', font=dict(color='#BFBFBF')),
                            tickfont=dict(color='#BFBFBF'), 
                            overlaying='y',
                            side='right'
                        )
                        # =========================================================================
                        # --- 💡 END: SYNTAXERROR FIX 💡 ---
                        # =========================================================================
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # --- TAB 2: TECHNICALS DEEP DIVE ---
                with tab2:
                    st.subheader(f"Intraday Technical Signals ({TA_TIMEFRAME})")
                    st.caption(f"Note: These signals update every 5 minutes based on the *last closed* candle for stability.")
                    
                    if detailed_metrics['Technical_Status'] != "OK":
                        st.error(f"Technical Analysis is unavailable: {detailed_metrics['Technical_Status']}")
                    else:
                        t_cols = st.columns(4) 
                        
                        t_cols[0].metric(
                            label=f"Supertrend Signal ({TA_SUPERTREND_LEN}, {TA_SUPERTREND_MULT})",
                            value=get_signal_narrative_binary(detailed_metrics['Supertrend_Signal']),
                            delta=f"Value: {detailed_metrics['Supertrend_Value']:.2f}"
                        )
                        t_cols[1].metric(
                            label=f"EMA Signal ({TA_EMA_LEN})",
                            value=get_signal_narrative_binary(detailed_metrics['EMA_Signal']),
                            delta=f"EMA Value: {detailed_metrics['EMA_Value']:.2f}"
                        )
                        t_cols[2].metric(
                            label=f"MACD Signal",
                            value=get_signal_narrative_binary(detailed_metrics['MACD_Signal']),
                            delta=f"MACDh: {detailed_metrics['MACD_Histogram']:.4f}"
                        )
                        t_cols[3].metric(
                            label=f"ADX/DI Signal ({TA_ADX_LEN})",
                            value=get_signal_narrative_binary(detailed_metrics['ADX_PDI_NPI_Signal']),
                            help="+DI vs -DI comparison."
                        )
                        
                        st.markdown("---")
                        st.subheader("Technicals Score Contribution")
                        
                        score_df = pd.DataFrame({
                            'Factor': ['Supertrend', 'EMA', 'MACD', 'ADX/DI'],
                            'Weight': [0.30, 0.25, 0.25, 0.20],
                            'Binary Score': [
                                detailed_metrics['Supertrend_Signal'],
                                detailed_metrics['EMA_Signal'],
                                detailed_metrics['MACD_Signal'],
                                detailed_metrics['ADX_PDI_NPI_Signal']
                            ],
                            'Contribution': [
                                detailed_metrics['Supertrend_Signal'] * 0.30,
                                detailed_metrics['EMA_Signal'] * 0.25,
                                detailed_metrics['MACD_Signal'] * 0.25,
                                detailed_metrics['ADX_PDI_NPI_Signal'] * 0.20
                            ]
                        })
                        st.dataframe(score_df, use_container_width=True, hide_index=True)

                # --- TAB 3: RAW DATA & METRICS ---
                with tab3:
                    st.subheader("Current Key Metrics")
                    d_cols = st.columns(3)
                    d_cols[0].metric("Total OI PCR (P/C)", f"{detailed_metrics['Total_PCR']:.3f}", help="Note: OI data can be delayed 3-5 mins.")
                    d_cols[1].metric("Near ATM Net OI Change %", f"{detailed_metrics['Near_OI_Net_Pct_Change']:.2f}%", help="Scale: +100 (Strong Put Add) to -100 (Strong Call Add). OI data can be delayed.")
                    d_cols[2].metric(f"Near ATM ({OI_SKEW_RANGE}pt) OI Skew", f"{detailed_metrics['Near_OI_Skew_Ratio']:.3f}", help="Puts/Calls OI in the skew range. OI data can be delayed.")
                    
                    p_cols = st.columns(4)
                    p_cols[0].metric("Nifty Spot Live Price", f"{detailed_metrics['Live_Price']:.2f}")
                    p_cols[1].metric("Nifty Futures VWAP", f"{detailed_metrics['VWAP']:.2f}")
                    p_cols[2].metric("Futures Premium", f"{detailed_metrics['Futures_Premium']:.2f} Pts")
                    p_cols[3].metric("Price vs VWAP %", f"{detailed_metrics['Price_VWAP_Pct']:.2f}%")
                            
                    st.markdown("---")
                    
                    st.subheader("Historical Metric Charts (Full Day)")
                    
                    chart_col1, chart_col2 = st.columns(2)
                    
                    with chart_col1:
                        fig_pcr = create_metric_chart(chart_data, 'Total_PCR', 'Total OI PCR (P/C) History', '#00C0F0')
                        st.plotly_chart(fig_pcr, use_container_width=True)
                        
                        fig_oi_change = create_metric_chart(chart_data, 'Near_OI_Net_Pct_Change', 'Near ATM Net OI Change % History', '#28C76F')
                        fig_oi_change.update_layout(yaxis=dict(range=[-100, 100]))
                        st.plotly_chart(fig_oi_change, use_container_width=True)

                        fig_vwap = create_metric_chart(chart_data, 'Price_VWAP_Pct', 'Price vs VWAP (%) History', '#FF7043')
                        st.plotly_chart(fig_vwap, use_container_width=True)
                        
                    with chart_col2:
                        fig_oi_skew = create_metric_chart(chart_data, 'Near_OI_Skew_Ratio', 'Near ATM OI Skew Ratio History', '#00C0F0')
                        st.plotly_chart(fig_oi_skew, use_container_width=True)

                        fig_premium = create_metric_chart(chart_data, 'Futures_Premium', 'Futures Premium (Pts) History', '#F3CA10')
                        st.plotly_chart(fig_premium, use_container_width=True)
                        
                        fig_open = create_metric_chart(chart_data, 'Price_Open_Pct', 'Price vs Open (%) History', '#FF7043')
                        st.plotly_chart(fig_open, use_container_width=True)

                    
                    st.markdown("---")
                    st.subheader("Score History Table")
                    
                    display_columns = {
                        'Timestamp_Str': 'Time',
                        'Final_Score': 'Total Score',
                        'Live_Price': 'Price',
                        'Total_PCR': 'PCR',
                        'Near_OI_Net_Pct_Change': 'Net OI %',
                        'Price_VWAP_Pct': 'VWAP %',
                        'Price_Volume_Score': 'Q1 Score',
                        'Derivatives_Score': 'Q2 Score',
                        'Technicals_Score': 'Q3 Score'
                    }
                    display_df = chart_data[list(display_columns.keys())].rename(columns=display_columns)
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

                # 3. FOOTER
                st.caption(f"Last updated: {detailed_metrics['Timestamp'].strftime('%I:%M:%S %p')} | Price/OI refresh every {REFRESH_INTERVAL_SECONDS}s | Technicals refresh every 5 mins.")

    except Exception as e:
        with placeholder.container():
            st.error(f"A critical error occurred: {e}", icon="🔥")
            st.exception(e) # Log the full stack trace
            
    time.sleep(REFRESH_INTERVAL_SECONDS)
