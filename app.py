import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="Crypto TSA Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

ACCENT  = '#e05c3a'
ACCENT2 = '#1a73e8'
ACCENT3 = '#1e8c3a'
ACCENT4 = '#7c3aed'

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; background-color: #f6f8fa; color: #1f2328; }
  .stApp { background-color: #f6f8fa; }
  section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #d0d7de; }
  .metric-card { background: #ffffff; border: 1px solid #d0d7de; border-radius: 8px; padding: 16px 20px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
  .metric-card .label { font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: #57606a; font-family: 'JetBrains Mono', monospace; }
  .metric-card .value { font-size: 26px; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #1a73e8; margin-top: 4px; }
  .metric-card .delta { font-size: 12px; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
  .section-header { font-family: 'JetBrains Mono', monospace; font-size: 13px; letter-spacing: 0.1em; color: #e05c3a; text-transform: uppercase; border-bottom: 1px solid #d0d7de; padding-bottom: 8px; margin-bottom: 16px; }
  h1 { font-family: 'JetBrains Mono', monospace; color: #1f2328 !important; }
  h2, h3 { font-family: 'Space Grotesk', sans-serif; color: #1f2328 !important; }
</style>
""", unsafe_allow_html=True)

def pl(**extra):
    """Build a plotly layout dict — extra kwargs override base values safely."""
    layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#ffffff',
        font=dict(family='JetBrains Mono, monospace', color='#1f2328', size=11),
        xaxis=dict(gridcolor='#eaeef2', linecolor='#d0d7de', tickcolor='#57606a'),
        yaxis=dict(gridcolor='#eaeef2', linecolor='#d0d7de', tickcolor='#57606a'),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#d0d7de', borderwidth=1),
    )
    layout.update(extra)
    return layout

def add_features(group):
    g = group.copy().sort_values('Date')
    g['Returns']       = g['Close'].pct_change()
    g['Log_Returns']   = np.log(g['Close'] / g['Close'].shift(1))
    g['MA_7']          = g['Close'].rolling(7).mean()
    g['MA_30']         = g['Close'].rolling(30).mean()
    g['MA_90']         = g['Close'].rolling(90).mean()
    g['Volatility_30'] = g['Log_Returns'].rolling(30).std() * np.sqrt(365)
    g['BB_Mid']        = g['Close'].rolling(20).mean()
    g['BB_Upper']      = g['BB_Mid'] + 2 * g['Close'].rolling(20).std()
    g['BB_Lower']      = g['BB_Mid'] - 2 * g['Close'].rolling(20).std()
    delta = g['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    g['RSI_14']      = 100 - (100 / (1 + rs))
    ema12            = g['Close'].ewm(span=12, adjust=False).mean()
    ema26            = g['Close'].ewm(span=26, adjust=False).mean()
    g['MACD']        = ema12 - ema26
    g['MACD_Signal'] = g['MACD'].ewm(span=9, adjust=False).mean()
    g['MACD_Hist']   = g['MACD'] - g['MACD_Signal']
    hl  = g['High'] - g['Low']
    hpc = (g['High'] - g['Close'].shift(1)).abs()
    lpc = (g['Low']  - g['Close'].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    g['ATR_14']         = tr.rolling(14).mean()
    g['Pct_Change_7d']  = g['Close'].pct_change(7)  * 100
    g['Pct_Change_30d'] = g['Close'].pct_change(30) * 100
    return g

@st.cache_data
def load_data():
    if os.path.exists("outputs/featured_crypto.csv"):
        df = pd.read_csv("outputs/featured_crypto.csv", parse_dates=['Date'])
    elif os.path.exists("master_crypto.csv"):
        df = pd.read_csv("master_crypto.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(['Name', 'Date']).reset_index(drop=True)
        df = df.groupby('Name', group_keys=False).apply(add_features).reset_index(drop=True)
    else:
        st.error("Data file not found. Place master_crypto.csv in the same folder as app.py.")
        st.stop()
    return df

@st.cache_data
def load_model_outputs():
    out = {}
    for key, path in {
        'arima_sarima': 'outputs/powerbi_arima_sarima.csv',
        'prophet':      'outputs/powerbi_prophet.csv',
        'lstm':         'outputs/powerbi_lstm.csv',
        'eval':         'outputs/powerbi_model_evaluation.csv',
        'stationarity': 'outputs/stationarity_tests.csv',
    }.items():
        if os.path.exists(path):
            tmp = pd.read_csv(path)
            for col in tmp.columns:
                if col.lower() in ('date', 'ds'):
                    try:
                        tmp[col] = pd.to_datetime(tmp[col])
                    except Exception:
                        pass
            out[key] = tmp
    return out

with st.spinner("Loading data..."):
    df = load_data()
    model_outputs = load_model_outputs()

all_coins = sorted(df['Name'].unique().tolist())
coin_colors = {
    'Bitcoin':  ACCENT2, 'Ethereum': ACCENT,   'Litecoin': ACCENT3,
    'XRP':      ACCENT4, 'Dogecoin': '#ffa657', 'Monero':   '#ff7b72',
    'Solana':   '#58a6ff','Cardano':  '#bc8cff', 'Stellar':  '#3fb950',
    'Chainlink':'#e3b341',
}
def get_color(coin):
    return coin_colors.get(coin, ACCENT2)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:700;
                color:#e05c3a;letter-spacing:0.05em;padding-bottom:12px;
                border-bottom:1px solid #d0d7de;margin-bottom:16px;">
        📈 CRYPTO TSA
    </div>
""", unsafe_allow_html=True)

    primary_coin  = st.selectbox("Primary Coin", all_coins,
                                  index=all_coins.index('Bitcoin') if 'Bitcoin' in all_coins else 0)
    compare_coins = st.multiselect("Compare Coins",
                                    [c for c in all_coins if c != primary_coin],
                                    default=[c for c in ['Ethereum','Litecoin','XRP','Dogecoin'] if c in all_coins])
    date_min   = df['Date'].min().date()
    date_max   = df['Date'].max().date()
    date_range = st.slider("Date Range", min_value=date_min, max_value=date_max, value=(date_min, date_max))
    st.markdown("---")
    st.markdown(f"""<div style="font-size:10px;color:#57606a;font-family:'JetBrains Mono',monospace;line-height:1.8;">
        <b style="color:#1f2328;">Dataset</b><br>
        Coins: {df['Name'].nunique()}<br>Rows: {len(df):,}<br>
        From: {df['Date'].min().date()}<br>To: {df['Date'].max().date()}
    </div>""", unsafe_allow_html=True)

mask       = (df['Date'].dt.date >= date_range[0]) & (df['Date'].dt.date <= date_range[1])
dff        = df[mask].copy()
primary_df = dff[dff['Name'] == primary_coin].sort_values('Date')

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:baseline;gap:16px;margin-bottom:4px;">
  <h1 style="font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:700;color:#1f2328;margin:0;">{primary_coin}</h1>
  <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#57606a;letter-spacing:0.1em;">TIME SERIES ANALYSIS</span>
</div>""", unsafe_allow_html=True)

# ── KPI cards ─────────────────────────────────────────────────────────────────
if len(primary_df) > 0:
    latest = primary_df.iloc[-1]
    prev   = primary_df.iloc[-2] if len(primary_df) > 1 else latest
    k1, k2, k3, k4, k5 = st.columns(5)

    def kpi(col, label, value, delta=None, pos=True):
        dc = ACCENT3 if pos else ACCENT
        dh = f'<div class="delta" style="color:{dc};">{delta}</div>' if delta else ''
        col.markdown(f'<div class="metric-card"><div class="label">{label}</div>'
                     f'<div class="value">{value}</div>{dh}</div>', unsafe_allow_html=True)

    pd_delta = latest['Close'] - prev['Close']
    pd_pct   = (pd_delta / prev['Close']) * 100
    kpi(k1, "Close Price", f"${latest['Close']:,.2f}", f"{pd_pct:+.2f}%", pd_delta >= 0)
    vol = latest['Volume']
    kpi(k2, "Volume", f"${vol/1e9:.2f}B" if vol > 1e9 else f"${vol/1e6:.1f}M")
    rsi = latest.get('RSI_14', np.nan)
    lbl = "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
    kpi(k3, f"RSI 14 · {lbl}", f"{rsi:.1f}" if not np.isnan(rsi) else "N/A")
    v30 = latest.get('Volatility_30', np.nan)
    kpi(k4, "Volatility 30d", f"{v30:.2f}" if not np.isnan(v30) else "N/A")
    p7  = latest.get('Pct_Change_7d', 0)
    kpi(k5, "7d Change", f"{p7:+.2f}%", None, p7 >= 0)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊  Price & Indicators", "🔀  Multi-Coin", "📉  Volatility & Returns",
    "🔮  Forecasting", "📋  Model Evaluation", "🗂️  Raw Data",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Price & Indicators
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f'<div class="section-header">{primary_coin} — Price History & Technical Indicators</div>',
                unsafe_allow_html=True)
    show_ma = st.checkbox("Moving Averages (7/30/90)", value=True)
    show_bb = st.checkbox("Bollinger Bands", value=True)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.55, 0.22, 0.23],
                        subplot_titles=('Price', 'Volume', 'RSI (14)'))
    fig.add_trace(go.Candlestick(
        x=primary_df['Date'], open=primary_df['Open'], high=primary_df['High'],
        low=primary_df['Low'], close=primary_df['Close'], name='OHLC',
        increasing_line_color=ACCENT3, decreasing_line_color=ACCENT,
        increasing_fillcolor=ACCENT3, decreasing_fillcolor=ACCENT), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=primary_df['Date'], y=primary_df['BB_Upper'],
            line=dict(color='rgba(26,115,232,0.4)', width=0.8), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=primary_df['Date'], y=primary_df['BB_Lower'],
            line=dict(color='rgba(26,115,232,0.4)', width=0.8), name='BB Lower',
            fill='tonexty', fillcolor='rgba(26,115,232,0.06)'), row=1, col=1)
    if show_ma:
        for ma, color, w in [('MA_7', ACCENT, 0.9), ('MA_30', ACCENT3, 1.1), ('MA_90', ACCENT4, 1.3)]:
            fig.add_trace(go.Scatter(x=primary_df['Date'], y=primary_df[ma],
                line=dict(color=color, width=w, dash='dot'), name=ma), row=1, col=1)
    vol_colors = [ACCENT3 if c >= o else ACCENT for c, o in zip(primary_df['Close'], primary_df['Open'])]
    fig.add_trace(go.Bar(x=primary_df['Date'], y=primary_df['Volume'],
        marker_color=vol_colors, opacity=0.6, name='Volume', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=primary_df['Date'], y=primary_df['RSI_14'],
        line=dict(color=ACCENT4, width=1.2), name='RSI 14', showlegend=False), row=3, col=1)
    fig.add_hline(y=70, line=dict(color=ACCENT,  dash='dash', width=0.8), row=3, col=1)
    fig.add_hline(y=30, line=dict(color=ACCENT3, dash='dash', width=0.8), row=3, col=1)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#ffffff',
                      font=dict(family='JetBrains Mono, monospace', color='#1f2328', size=11),
                      margin=dict(l=40, r=20, t=50, b=40),
                      legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#d0d7de', borderwidth=1),
                      height=680, title=f'{primary_coin} — Candlestick Chart',
                      xaxis_rangeslider_visible=False)
    fig.update_xaxes(gridcolor='#eaeef2', linecolor='#d0d7de', tickcolor='#57606a')
    fig.update_yaxes(gridcolor='#eaeef2', linecolor='#d0d7de', tickcolor='#57606a')
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">MACD</div>', unsafe_allow_html=True)
    fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.4])
    fig_macd.add_trace(go.Scatter(x=primary_df['Date'], y=primary_df['Close'],
        line=dict(color=ACCENT2, width=1.2), name='Close'), row=1, col=1)
    fig_macd.add_trace(go.Scatter(x=primary_df['Date'], y=primary_df['MACD'],
        line=dict(color=ACCENT2, width=1.0), name='MACD'), row=2, col=1)
    fig_macd.add_trace(go.Scatter(x=primary_df['Date'], y=primary_df['MACD_Signal'],
        line=dict(color=ACCENT, width=1.0), name='Signal'), row=2, col=1)
    hist_colors = [ACCENT3 if v >= 0 else '#cf3828' for v in primary_df['MACD_Hist'].fillna(0)]
    fig_macd.add_trace(go.Bar(x=primary_df['Date'], y=primary_df['MACD_Hist'],
        marker_color=hist_colors, opacity=0.7, name='Histogram'), row=2, col=1)
    fig_macd.update_layout(**pl(height=420, title=f'{primary_coin} — MACD'))
    st.plotly_chart(fig_macd, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-Coin
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    all_selected = [primary_coin] + compare_coins

    st.markdown('<div class="section-header">Normalized Price Comparison (Base = 1)</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    for coin in all_selected:
        sub = dff[dff['Name'] == coin].sort_values('Date')
        sub = sub[sub['Close'] > 0]
        if len(sub) == 0: continue
        norm = sub['Close'] / sub['Close'].iloc[0]
        fig2.add_trace(go.Scatter(x=sub['Date'], y=norm, name=coin, mode='lines',
            line=dict(color=get_color(coin), width=1.4)))
    fig2.update_layout(**pl(height=420, yaxis_title='Normalized Price', hovermode='x unified'))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Price Correlation Heatmap</div>', unsafe_allow_html=True)
    pivot = dff.pivot_table(index='Date', columns='Name', values='Close')
    avail = [c for c in all_selected if c in pivot.columns]
    if len(avail) >= 2:
        corr = pivot[avail].corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale='RdYlGn', zmid=0,
            text=np.round(corr.values, 2), texttemplate='%{text}', showscale=True))
        fig_corr.update_layout(**pl(height=420, title='Return Correlation Matrix'))
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Select at least 2 coins to see correlation.")

    st.markdown('<div class="section-header">Volume Comparison</div>', unsafe_allow_html=True)
    fig_vol2 = go.Figure()
    for coin in all_selected:
        sub = dff[dff['Name'] == coin].sort_values('Date')
        fig_vol2.add_trace(go.Scatter(x=sub['Date'], y=sub['Volume'], name=coin,
            mode='lines', line=dict(color=get_color(coin), width=1.0),
            fill='tozeroy', fillcolor='rgba(26,115,232,0.07)'))
    fig_vol2.update_layout(**pl(height=350, yaxis_title='Volume', hovermode='x unified'))
    st.plotly_chart(fig_vol2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Volatility & Returns
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">30-Day Annualised Volatility</div>', unsafe_allow_html=True)
        fig_v = go.Figure()
        for coin in [primary_coin] + compare_coins[:5]:
            sub = dff[dff['Name'] == coin].sort_values('Date').dropna(subset=['Volatility_30'])
            fig_v.add_trace(go.Scatter(x=sub['Date'], y=sub['Volatility_30'], name=coin,
                line=dict(color=get_color(coin), width=1.2)))
        fig_v.update_layout(**pl(height=360, yaxis_title='Volatility (annualised)'))
        st.plotly_chart(fig_v, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Log Returns Distribution</div>', unsafe_allow_html=True)
        sub_ret = primary_df['Log_Returns'].dropna()
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(x=sub_ret, nbinsx=80,
            marker_color=get_color(primary_coin), opacity=0.8, name='Log Returns'))
        fig_ret.add_vline(x=float(sub_ret.mean()), line=dict(color='#1f2328', dash='dash', width=1))
        fig_ret.add_vline(x=float(sub_ret.mean() + 2*sub_ret.std()), line=dict(color=ACCENT, dash='dot', width=0.8))
        fig_ret.add_vline(x=float(sub_ret.mean() - 2*sub_ret.std()), line=dict(color=ACCENT, dash='dot', width=0.8))
        fig_ret.update_layout(**pl(height=360, xaxis_title='Log Return', yaxis_title='Count'))
        st.plotly_chart(fig_ret, use_container_width=True)

    st.markdown(f'<div class="section-header">{primary_coin} — Monthly Volatility Heatmap</div>', unsafe_allow_html=True)
    btc_vol = primary_df[['Date','Volatility_30']].dropna().copy()
    btc_vol['Year']  = btc_vol['Date'].dt.year
    btc_vol['Month'] = btc_vol['Date'].dt.month
    vol_pivot = btc_vol.groupby(['Year','Month'])['Volatility_30'].mean().unstack()
    if not vol_pivot.empty:
        fig_vh = go.Figure(go.Heatmap(
            z=vol_pivot.values,
            x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:vol_pivot.shape[1]],
            y=vol_pivot.index.tolist(),
            colorscale='RdYlGn_r',
            text=np.round(vol_pivot.values, 2), texttemplate='%{text}'))
        fig_vh.update_layout(**pl(height=380, title=f'{primary_coin} Average Monthly Volatility'))
        st.plotly_chart(fig_vh, use_container_width=True)

    st.markdown('<div class="section-header">RSI Comparison</div>', unsafe_allow_html=True)
    fig_rsi = go.Figure()
    for coin in [primary_coin] + compare_coins[:3]:
        sub = dff[dff['Name'] == coin].sort_values('Date').dropna(subset=['RSI_14'])
        fig_rsi.add_trace(go.Scatter(x=sub['Date'], y=sub['RSI_14'], name=coin,
            line=dict(color=get_color(coin), width=1.1)))
    fig_rsi.add_hline(y=70, line=dict(color=ACCENT,  dash='dash', width=0.8))
    fig_rsi.add_hline(y=30, line=dict(color=ACCENT3, dash='dash', width=0.8))
    fig_rsi.update_layout(**pl(height=340, yaxis_title='RSI (14)',
                               yaxis=dict(range=[0, 100], gridcolor='#21262d', tickcolor='#8b949e')))
    st.plotly_chart(fig_rsi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Forecasting
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Model Forecasts (Bitcoin · Pre-computed)</div>', unsafe_allow_html=True)
    if not model_outputs:
        st.info("No pre-computed forecasts found. Run crypto_tsa_project.py once to generate the outputs/ folder, then redeploy.")
    else:
        if 'arima_sarima' in model_outputs:
            st.markdown("#### ARIMA & SARIMA")
            as_df = model_outputs['arima_sarima']
            fig_as = go.Figure()
            fig_as.add_trace(go.Scatter(x=as_df['Date'], y=as_df['Actual'],
                line=dict(color=ACCENT2, width=1.5), name='Actual'))
            if 'ARIMA_Pred' in as_df.columns:
                fig_as.add_trace(go.Scatter(x=as_df['Date'], y=as_df['ARIMA_Pred'],
                    line=dict(color=ACCENT, width=1.5, dash='dash'), name='ARIMA'))
            if 'SARIMA_Pred' in as_df.columns:
                fig_as.add_trace(go.Scatter(x=as_df['Date'], y=as_df['SARIMA_Pred'],
                    line=dict(color=ACCENT3, width=1.5, dash='dot'), name='SARIMA'))
            fig_as.update_layout(**pl(height=380, yaxis_title='Price (USD)',
                                      title='Bitcoin — ARIMA & SARIMA Forecast vs Actual'))
            st.plotly_chart(fig_as, use_container_width=True)

        if 'prophet' in model_outputs:
            st.markdown("#### Prophet")
            pr_df = model_outputs['prophet']
            pr_df.columns = [c.strip() for c in pr_df.columns]
            fig_pr = go.Figure()
            if 'yhat_lower' in pr_df.columns and 'yhat_upper' in pr_df.columns:
                fig_pr.add_trace(go.Scatter(
                    x=pd.concat([pr_df['ds'], pr_df['ds'][::-1]]),
                    y=pd.concat([pr_df['yhat_upper'], pr_df['yhat_lower'][::-1]]),
                    fill='toself', fillcolor='rgba(124,58,237,0.10)',
                    line=dict(color='rgba(0,0,0,0)'), name='Uncertainty Band'))
            fig_pr.add_trace(go.Scatter(x=pr_df['ds'], y=pr_df['yhat'],
                line=dict(color=ACCENT4, width=1.5), name='Prophet Forecast'))
            fig_pr.update_layout(**pl(height=380, yaxis_title='Price (USD)',
                                      title='Bitcoin — Prophet Forecast (+90 days)'))
            st.plotly_chart(fig_pr, use_container_width=True)

        if 'lstm' in model_outputs:
            st.markdown("#### LSTM")
            lstm_df = model_outputs['lstm']
            fig_lstm = go.Figure()
            fig_lstm.add_trace(go.Scatter(x=lstm_df['Date'], y=lstm_df['Actual'],
                line=dict(color=ACCENT2, width=1.5), name='Actual'))
            if 'LSTM_Pred' in lstm_df.columns:
                fig_lstm.add_trace(go.Scatter(x=lstm_df['Date'], y=lstm_df['LSTM_Pred'],
                    line=dict(color=ACCENT, width=1.5, dash='dash'), name='LSTM Predicted'))
            fig_lstm.update_layout(**pl(height=380, yaxis_title='Price (USD)',
                                        title='Bitcoin — LSTM Forecast vs Actual'))
            st.plotly_chart(fig_lstm, use_container_width=True)

        if 'stationarity' in model_outputs:
            st.markdown("#### ADF Stationarity Tests")
            adf = model_outputs['stationarity'].copy()
            if 'Stationary' in adf.columns:
                adf['Result'] = adf['Stationary'].map({True: '✓ Stationary', False: '✗ Non-stationary'})
            st.dataframe(adf, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Model Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Model Evaluation — Bitcoin Price Forecasting</div>', unsafe_allow_html=True)
    if 'eval' in model_outputs:
        eval_df = model_outputs['eval']
        metrics = [c for c in ['MAE', 'RMSE', 'MAPE_%'] if c in eval_df.columns]
        if metrics:
            bar_colors = [ACCENT, ACCENT2, ACCENT3, ACCENT4]
            cols = st.columns(len(metrics))
            for col, metric in zip(cols, metrics):
                fig_bar = go.Figure(go.Bar(
                    x=eval_df['Model'],
                    y=eval_df[metric],
                    marker_color=bar_colors[:len(eval_df)],
                    text=[f"{v:,.0f}" if metric != 'MAPE_%' else f"{v:.1f}%" for v in eval_df[metric]],
                    textposition='outside',
                    textfont=dict(color='#1f2328', size=10)
                ))
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='#ffffff',
                    font=dict(family='JetBrains Mono, monospace', color='#1f2328', size=11),
                    margin=dict(l=40, r=20, t=50, b=40),
                    xaxis=dict(gridcolor='#eaeef2', linecolor='#d0d7de', tickcolor='#57606a', tickangle=-20),
                    yaxis=dict(gridcolor='#eaeef2', linecolor='#d0d7de', tickcolor='#57606a'),
                    legend=dict(bgcolor='rgba(255,255,255,0.9)', bordercolor='#d0d7de', borderwidth=1),
                    height=320, title=metric, showlegend=False
                )
                col.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### Leaderboard")
        display_df = eval_df.copy()
        if 'MAPE_%' in display_df.columns:
            display_df = display_df.sort_values('MAPE_%')
            display_df.insert(0, 'Rank', range(1, len(display_df)+1))
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run crypto_tsa_project.py first to generate model evaluation outputs.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Raw Data
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-header">Dataset Explorer</div>', unsafe_allow_html=True)
    selected_coins_raw = st.multiselect("Filter by Coin", all_coins,
                                         default=[primary_coin], key='raw_coin_filter')
    default_cols = [c for c in ['Date','Name','Open','High','Low','Close','Volume',
                                 'MA_7','MA_30','RSI_14','MACD','Volatility_30'] if c in dff.columns]
    show_cols = st.multiselect("Columns", dff.columns.tolist(),
                                default=default_cols, key='raw_col_filter')
    filtered_raw = dff[dff['Name'].isin(selected_coins_raw)][show_cols] if selected_coins_raw else dff[show_cols]
    st.markdown(f"<div style='font-size:11px;color:#57606a;font-family:JetBrains Mono,monospace;margin-bottom:8px;'>"
                f"{len(filtered_raw):,} rows · {len(show_cols)} columns</div>", unsafe_allow_html=True)
    st.dataframe(filtered_raw.sort_values('Date', ascending=False).head(500),
                 use_container_width=True, hide_index=True)
    csv_data = filtered_raw.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download CSV", csv_data,
                        file_name=f"crypto_filtered_{primary_coin}.csv", mime="text/csv")
