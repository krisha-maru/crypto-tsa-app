# =============================================================================
# TIME SERIES ANALYSIS WITH CRYPTOCURRENCY — END TO END
# Amdox Technologies | Data Analytics Project
# Dataset: master_crypto.csv (23 coins, 2013–2021)
# =============================================================================
# INSTALL (run once in terminal):
# pip install pandas numpy matplotlib seaborn plotly statsmodels pmdarima
#             prophet scikit-learn torch scipy
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# ── Output folder (all CSVs for Power BI go here) ────────────────────────────
os.makedirs("outputs", exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#c9d1d9',
    'grid.color':       '#21262d',
    'grid.linewidth':   0.5,
    'font.family':      'monospace',
})
ACCENT   = '#f78166'
ACCENT2  = '#79c0ff'
ACCENT3  = '#56d364'
ACCENT4  = '#d2a8ff'

# =============================================================================
# SECTION 1 — DATA LOADING & PREPROCESSING
# =============================================================================
print("=" * 60)
print("SECTION 1 — DATA LOADING & PREPROCESSING")
print("=" * 60)

df = pd.read_csv("master_crypto.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Name', 'Date']).reset_index(drop=True)

print(f"Shape         : {df.shape}")
print(f"Coins         : {df['Name'].nunique()}")
print(f"Date range    : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Null values   : {df.isnull().sum().sum()}")
print()

# ── Feature Engineering ──────────────────────────────────────────────────────
def add_features(group):
    g = group.copy().sort_values('Date')

    # Returns
    g['Returns']     = g['Close'].pct_change()
    g['Log_Returns'] = np.log(g['Close'] / g['Close'].shift(1))

    # Moving Averages
    g['MA_7']  = g['Close'].rolling(7).mean()
    g['MA_30'] = g['Close'].rolling(30).mean()
    g['MA_90'] = g['Close'].rolling(90).mean()

    # Volatility (30-day rolling std of log returns)
    g['Volatility_30'] = g['Log_Returns'].rolling(30).std() * np.sqrt(365)

    # Bollinger Bands (20-day)
    g['BB_Mid']   = g['Close'].rolling(20).mean()
    g['BB_Upper'] = g['BB_Mid'] + 2 * g['Close'].rolling(20).std()
    g['BB_Lower'] = g['BB_Mid'] - 2 * g['Close'].rolling(20).std()

    # RSI (14-day)
    delta = g['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    g['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12       = g['Close'].ewm(span=12, adjust=False).mean()
    ema26       = g['Close'].ewm(span=26, adjust=False).mean()
    g['MACD']         = ema12 - ema26
    g['MACD_Signal']  = g['MACD'].ewm(span=9, adjust=False).mean()
    g['MACD_Hist']    = g['MACD'] - g['MACD_Signal']

    # ATR (Average True Range — volatility proxy)
    hl  = g['High'] - g['Low']
    hpc = (g['High'] - g['Close'].shift(1)).abs()
    lpc = (g['Low']  - g['Close'].shift(1)).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    g['ATR_14'] = tr.rolling(14).mean()

    # Price change %
    g['Pct_Change_7d']  = g['Close'].pct_change(7)  * 100
    g['Pct_Change_30d'] = g['Close'].pct_change(30) * 100

    return g

df = df.groupby('Name', group_keys=False).apply(add_features)
df = df.reset_index(drop=True)

print(f"Features added. New shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ── Save Power BI master file ─────────────────────────────────────────────────
df.to_csv("outputs/featured_crypto.csv", index=False)
print("\n✓ Saved: outputs/featured_crypto.csv  (use this in Power BI)")


# =============================================================================
# SECTION 2 — EXPLORATORY DATA ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2 — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── Plot 1: BTC Full Price History ───────────────────────────────────────────
btc = df[df['Name'] == 'Bitcoin'].dropna(subset=['MA_30'])

fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
fig.suptitle('Bitcoin — Full Price History & Indicators', fontsize=16, color=ACCENT, fontweight='bold', y=1.01)

ax1 = axes[0]
ax1.plot(btc['Date'], btc['Close'],  color=ACCENT2,  lw=1.2, label='Close', alpha=0.9)
ax1.plot(btc['Date'], btc['MA_7'],   color=ACCENT,   lw=1.0, linestyle='--', label='MA 7',  alpha=0.8)
ax1.plot(btc['Date'], btc['MA_30'],  color=ACCENT3,  lw=1.0, linestyle='--', label='MA 30', alpha=0.8)
ax1.plot(btc['Date'], btc['MA_90'],  color=ACCENT4,  lw=1.0, linestyle='--', label='MA 90', alpha=0.8)
ax1.fill_between(btc['Date'], btc['BB_Lower'], btc['BB_Upper'], alpha=0.08, color=ACCENT2, label='Bollinger Bands')
ax1.set_ylabel('Price (USD)')
ax1.legend(fontsize=8, loc='upper left')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.bar(btc['Date'], btc['Volume'], color=ACCENT2, alpha=0.5, width=1)
ax2.set_ylabel('Volume')
ax2.grid(True, alpha=0.3)

ax3 = axes[2]
ax3.plot(btc['Date'], btc['RSI_14'], color=ACCENT4, lw=1.0)
ax3.axhline(70, color=ACCENT,  linestyle='--', lw=0.8, alpha=0.7, label='Overbought (70)')
ax3.axhline(30, color=ACCENT3, linestyle='--', lw=0.8, alpha=0.7, label='Oversold (30)')
ax3.fill_between(btc['Date'], btc['RSI_14'], 70, where=(btc['RSI_14'] >= 70), color=ACCENT,  alpha=0.2)
ax3.fill_between(btc['Date'], btc['RSI_14'], 30, where=(btc['RSI_14'] <= 30), color=ACCENT3, alpha=0.2)
ax3.set_ylabel('RSI (14)')
ax3.set_ylim(0, 100)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig('outputs/01_btc_price_history.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot 1 saved: BTC price history + RSI + Volume")

# ── Plot 2: Multi-coin price comparison (normalized) ─────────────────────────
top_coins = ['Bitcoin', 'Ethereum', 'Litecoin', 'XRP', 'Dogecoin', 'Monero']
colors    = [ACCENT, ACCENT2, ACCENT3, ACCENT4, '#ffa657', '#ff7b72']

fig, ax = plt.subplots(figsize=(16, 7))
fig.suptitle('Normalized Price Comparison — Top Coins (Base = 1)', fontsize=14, color=ACCENT, fontweight='bold')

for coin, color in zip(top_coins, colors):
    sub = df[df['Name'] == coin].sort_values('Date')
    sub = sub[sub['Close'] > 0]
    normalized = sub['Close'] / sub['Close'].iloc[0]
    ax.plot(sub['Date'], normalized, label=coin, color=color, lw=1.2)

ax.set_ylabel('Normalized Price')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig('outputs/02_normalized_price_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot 2 saved: Normalized price comparison")

# ── Plot 3: Correlation Heatmap ──────────────────────────────────────────────
pivot = df.pivot_table(index='Date', columns='Name', values='Close')
pivot = pivot[['Bitcoin','Ethereum','Litecoin','XRP','Dogecoin','Monero','Stellar','Cardano','Chainlink','Solana']]
corr  = pivot.corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=ax, linewidths=0.5, linecolor='#21262d',
            cbar_kws={'shrink': 0.8}, annot_kws={'size': 8})
ax.set_title('Price Correlation Heatmap — Top 10 Coins', fontsize=14, color=ACCENT, pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot 3 saved: Correlation heatmap")

# Save correlation for Power BI
corr.reset_index().to_csv('outputs/powerbi_correlation.csv', index=False)

# ── Plot 4: Volatility comparison ────────────────────────────────────────────
vol_coins = ['Bitcoin', 'Ethereum', 'Dogecoin', 'XRP', 'Solana', 'Cardano']
vol_data  = df[df['Name'].isin(vol_coins)][['Name', 'Date', 'Volatility_30']].dropna()

fig, ax = plt.subplots(figsize=(16, 6))
for coin, color in zip(vol_coins, colors):
    sub = vol_data[vol_data['Name'] == coin]
    ax.plot(sub['Date'], sub['Volatility_30'], label=coin, lw=1.0, alpha=0.85)

ax.set_title('30-Day Rolling Annualised Volatility', fontsize=14, color=ACCENT, fontweight='bold')
ax.set_ylabel('Volatility (annualised)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig('outputs/04_volatility_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot 4 saved: Volatility comparison")

# ── Plot 5: Returns Distribution ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle('Daily Log Returns Distribution', fontsize=14, color=ACCENT, fontweight='bold')

for ax, (coin, color) in zip(axes.flatten(), zip(vol_coins, colors)):
    sub = df[df['Name'] == coin]['Log_Returns'].dropna()
    ax.hist(sub, bins=80, color=color, alpha=0.75, edgecolor='none')
    ax.axvline(sub.mean(), color='white', linestyle='--', lw=1, label=f'μ={sub.mean():.4f}')
    ax.axvline(sub.mean() + 2*sub.std(), color=ACCENT,  linestyle=':', lw=0.8)
    ax.axvline(sub.mean() - 2*sub.std(), color=ACCENT,  linestyle=':', lw=0.8)
    ax.set_title(coin, fontsize=10, color=color)
    ax.set_xlabel('Log Return')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/05_returns_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot 5 saved: Returns distribution")

# ── Plot 6: MACD for BTC ─────────────────────────────────────────────────────
btc_macd = btc.tail(365)   # last year only for clarity

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [2, 1]})
fig.suptitle('Bitcoin — MACD Indicator (Last 365 Days)', fontsize=14, color=ACCENT, fontweight='bold')

ax1.plot(btc_macd['Date'], btc_macd['Close'], color=ACCENT2, lw=1.2, label='BTC Close')
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(btc_macd['Date'], btc_macd['MACD'],        color=ACCENT2,  lw=1.0, label='MACD')
ax2.plot(btc_macd['Date'], btc_macd['MACD_Signal'],  color=ACCENT,   lw=1.0, label='Signal')
colors_hist = [ACCENT3 if v >= 0 else '#ff7b72' for v in btc_macd['MACD_Hist']]
ax2.bar(btc_macd['Date'], btc_macd['MACD_Hist'], color=colors_hist, alpha=0.6, width=1)
ax2.set_ylabel('MACD')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.tight_layout()
plt.savefig('outputs/06_macd_btc.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot 6 saved: MACD")


# =============================================================================
# SECTION 3 — STATIONARITY TESTS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3 — STATIONARITY TESTS (ADF)")
print("=" * 60)

from statsmodels.tsa.stattools import adfuller

def adf_test(series, name):
    result = adfuller(series.dropna(), autolag='AIC')
    stat, p, lags = result[0], result[1], result[2]
    stat_flag = "✓ Stationary" if p < 0.05 else "✗ Non-stationary"
    print(f"  {name:<30} | ADF={stat:7.3f} | p={p:.4f} | {stat_flag}")
    return {'Series': name, 'ADF Stat': round(stat,4), 'p-value': round(p,4), 'Stationary': p < 0.05}

results = []
for coin in ['Bitcoin', 'Ethereum', 'Litecoin', 'XRP', 'Dogecoin']:
    sub = df[df['Name'] == coin].sort_values('Date')
    results.append(adf_test(sub['Close'],       f"{coin} Close Price"))
    results.append(adf_test(sub['Log_Returns'], f"{coin} Log Returns"))

adf_df = pd.DataFrame(results)
adf_df.to_csv('outputs/stationarity_tests.csv', index=False)
print("\n✓ Saved: outputs/stationarity_tests.csv")
print("  → Close prices are NON-STATIONARY (expected)")
print("  → Log returns are STATIONARY (use for ARIMA)")


# =============================================================================
# SECTION 4 — ARIMA / SARIMA FORECASTING (Bitcoin)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4 — ARIMA/SARIMA (Bitcoin)")
print("=" * 60)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── ACF / PACF plot ───────────────────────────────────────────────────────────
btc_returns = df[df['Name']=='Bitcoin']['Log_Returns'].dropna()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
fig.suptitle('Bitcoin Log Returns — ACF & PACF', fontsize=13, color=ACCENT, fontweight='bold')
plot_acf(btc_returns,  ax=ax1, lags=40, color=ACCENT2, title='ACF')
plot_pacf(btc_returns, ax=ax2, lags=40, color=ACCENT,  title='PACF')
ax1.set_facecolor('#161b22'); ax2.set_facecolor('#161b22')
plt.tight_layout()
plt.savefig('outputs/07_acf_pacf.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot 7 saved: ACF/PACF")

# ── Train/Test split ─────────────────────────────────────────────────────────
btc_close = df[df['Name']=='Bitcoin'][['Date','Close']].set_index('Date').sort_index()
btc_close = btc_close[btc_close.index >= '2017-01-01']   # use recent 4.5 years

split_idx = int(len(btc_close) * 0.8)
train = btc_close.iloc[:split_idx]
test  = btc_close.iloc[split_idx:]
print(f"Train: {len(train)} rows | Test: {len(test)} rows")

# ── ARIMA ─────────────────────────────────────────────────────────────────────
print("\nFitting ARIMA(2,1,2)...")
arima_model = ARIMA(train['Close'], order=(2, 1, 2))
arima_fit   = arima_model.fit()
arima_pred  = arima_fit.forecast(steps=len(test))
arima_pred.index = test.index

arima_mae  = mean_absolute_error(test['Close'], arima_pred)
arima_rmse = np.sqrt(mean_squared_error(test['Close'], arima_pred))
arima_mape = np.mean(np.abs((test['Close'].values - arima_pred.values) / test['Close'].values)) * 100
print(f"ARIMA  → MAE: {arima_mae:,.0f} | RMSE: {arima_rmse:,.0f} | MAPE: {arima_mape:.2f}%")

# ── SARIMA ────────────────────────────────────────────────────────────────────
print("Fitting SARIMA(2,1,2)(1,0,1,7)...")
sarima_model = SARIMAX(train['Close'], order=(2,1,2), seasonal_order=(1,0,1,7))
sarima_fit   = sarima_model.fit(disp=False)
sarima_pred  = sarima_fit.forecast(steps=len(test))
sarima_pred.index = test.index

sarima_mae  = mean_absolute_error(test['Close'], sarima_pred)
sarima_rmse = np.sqrt(mean_squared_error(test['Close'], sarima_pred))
sarima_mape = np.mean(np.abs((test['Close'].values - sarima_pred.values) / test['Close'].values)) * 100
print(f"SARIMA → MAE: {sarima_mae:,.0f} | RMSE: {sarima_rmse:,.0f} | MAPE: {sarima_mape:.2f}%")

# ── Plot ARIMA vs SARIMA ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(train.index, train['Close'], color='#8b949e', lw=0.8, label='Train')
ax.plot(test.index,  test['Close'],  color=ACCENT2,  lw=1.5, label='Actual')
ax.plot(test.index,  arima_pred,     color=ACCENT,   lw=1.5, linestyle='--', label=f'ARIMA  (MAPE={arima_mape:.1f}%)')
ax.plot(test.index,  sarima_pred,    color=ACCENT3,  lw=1.5, linestyle=':',  label=f'SARIMA (MAPE={sarima_mape:.1f}%)')
ax.axvline(test.index[0], color='white', linestyle='--', lw=0.8, alpha=0.5, label='Train/Test split')
ax.set_title('Bitcoin — ARIMA & SARIMA Forecast vs Actual', fontsize=14, color=ACCENT, fontweight='bold')
ax.set_ylabel('Price (USD)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()
plt.savefig('outputs/08_arima_sarima_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot 8 saved: ARIMA/SARIMA forecast")

# Save predictions for Power BI
arima_out = pd.DataFrame({
    'Date': test.index, 'Actual': test['Close'].values,
    'ARIMA_Pred': arima_pred.values, 'SARIMA_Pred': sarima_pred.values
})
arima_out.to_csv('outputs/powerbi_arima_sarima.csv', index=False)


# =============================================================================
# SECTION 5 — PROPHET FORECASTING
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5 — PROPHET (Bitcoin)")
print("=" * 60)

try:
    from prophet import Prophet

    btc_prophet = df[df['Name']=='Bitcoin'][['Date','Close']].copy()
    btc_prophet.columns = ['ds', 'y']
    btc_prophet['ds'] = pd.to_datetime(btc_prophet['ds']).dt.tz_localize(None)
    btc_prophet = btc_prophet[btc_prophet['ds'] >= '2017-01-01'].reset_index(drop=True)

    split = int(len(btc_prophet) * 0.8)
    train_p = btc_prophet.iloc[:split]
    test_p  = btc_prophet.iloc[split:]

    prophet_model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative'
    )
    prophet_model.fit(train_p)

    future     = prophet_model.make_future_dataframe(periods=len(test_p) + 90)
    forecast   = prophet_model.predict(future)

    # Test metrics
    test_forecast = forecast[forecast['ds'].isin(test_p['ds'])]
    if len(test_forecast) > 0:
        prophet_mae  = mean_absolute_error(test_p['y'].values[:len(test_forecast)], test_forecast['yhat'].values)
        prophet_rmse = np.sqrt(mean_squared_error(test_p['y'].values[:len(test_forecast)], test_forecast['yhat'].values))
        prophet_mape = np.mean(np.abs((test_p['y'].values[:len(test_forecast)] - test_forecast['yhat'].values) /
                                       test_p['y'].values[:len(test_forecast)])) * 100
        print(f"Prophet → MAE: {prophet_mae:,.0f} | RMSE: {prophet_rmse:,.0f} | MAPE: {prophet_mape:.2f}%")

    # ── Prophet plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(btc_prophet['ds'], btc_prophet['y'], color='#8b949e', lw=0.8, label='Actual')
    ax.plot(forecast['ds'], forecast['yhat'], color=ACCENT4, lw=1.5, label='Prophet Forecast')
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                    alpha=0.15, color=ACCENT4, label='Uncertainty Band')
    ax.axvline(train_p['ds'].iloc[-1], color='white', linestyle='--', lw=0.8, alpha=0.5, label='Train/Test split')
    ax.set_title('Bitcoin — Prophet Forecast (+90 days into future)', fontsize=14, color=ACCENT, fontweight='bold')
    ax.set_ylabel('Price (USD)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig('outputs/09_prophet_forecast.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Plot 9 saved: Prophet forecast")

    # Prophet components
    fig2 = prophet_model.plot_components(forecast)
    fig2.savefig('outputs/10_prophet_components.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Plot 10 saved: Prophet components (trend + seasonality)")

    # Save for Power BI
    forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv('outputs/powerbi_prophet.csv', index=False)

except ImportError:
    print("⚠ Prophet not installed. Run: pip install prophet")
    prophet_mae = prophet_rmse = prophet_mape = None


# =============================================================================
# SECTION 6 — LSTM FORECASTING
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 6 — LSTM (Bitcoin)")
print("=" * 60)

try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler

    # ── Data prep ────────────────────────────────────────────────────────────
    btc_lstm = df[df['Name']=='Bitcoin'][['Date','Close']].sort_values('Date')
    btc_lstm = btc_lstm[btc_lstm['Date'] >= '2017-01-01'].reset_index(drop=True)

    scaler      = MinMaxScaler()
    scaled      = scaler.fit_transform(btc_lstm[['Close']])
    LOOKBACK    = 60
    FORECAST    = 1
    EPOCHS      = 60
    BATCH_SIZE  = 32

    def make_sequences(data, lookback):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback])
        return np.array(X), np.array(y)

    X, y = make_sequences(scaled, LOOKBACK)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t  = torch.FloatTensor(X_test)

    # ── Model ─────────────────────────────────────────────────────────────────
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.fc   = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1)
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model     = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # ── Training ──────────────────────────────────────────────────────────────
    losses = []
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    print(f"Training LSTM for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6f}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        lstm_pred_scaled = model(X_test_t).numpy()

    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
    y_test_actual = scaler.inverse_transform(y_test)

    lstm_mae  = mean_absolute_error(y_test_actual, lstm_pred)
    lstm_rmse = np.sqrt(mean_squared_error(y_test_actual, lstm_pred))
    lstm_mape = np.mean(np.abs((y_test_actual - lstm_pred) / y_test_actual)) * 100
    print(f"\nLSTM → MAE: {lstm_mae:,.0f} | RMSE: {lstm_rmse:,.0f} | MAPE: {lstm_mape:.2f}%")

    # ── Training loss plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, color=ACCENT4, lw=1.5)
    ax.set_title('LSTM Training Loss', fontsize=13, color=ACCENT, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/11_lstm_training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── LSTM Prediction plot ───────────────────────────────────────────────────
    test_dates = btc_lstm['Date'].values[LOOKBACK + split:]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(test_dates, y_test_actual, color=ACCENT2, lw=1.5, label='Actual')
    ax.plot(test_dates, lstm_pred,     color=ACCENT,  lw=1.5, linestyle='--',
            label=f'LSTM Predicted (MAPE={lstm_mape:.1f}%)')
    ax.set_title('Bitcoin — LSTM Forecast vs Actual', fontsize=14, color=ACCENT, fontweight='bold')
    ax.set_ylabel('Price (USD)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/12_lstm_forecast.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Plot 11-12 saved: LSTM loss + forecast")

    # Save for Power BI
    lstm_out = pd.DataFrame({
        'Date': test_dates,
        'Actual': y_test_actual.flatten(),
        'LSTM_Pred': lstm_pred.flatten()
    })
    lstm_out.to_csv('outputs/powerbi_lstm.csv', index=False)

except ImportError:
    print("⚠ PyTorch not installed. Run: pip install torch")
    lstm_mae = lstm_rmse = lstm_mape = None


# =============================================================================
# SECTION 7 — MODEL EVALUATION COMPARISON
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 7 — MODEL EVALUATION SUMMARY")
print("=" * 60)

eval_data = {
    'Model':  ['ARIMA(2,1,2)', 'SARIMA(2,1,2)(1,0,1,7)', 'Prophet', 'LSTM'],
    'MAE':    [arima_mae,   sarima_mae,   prophet_mae   if 'prophet_mae'   in dir() else None, lstm_mae   if 'lstm_mae'   in dir() else None],
    'RMSE':   [arima_rmse,  sarima_rmse,  prophet_rmse  if 'prophet_rmse'  in dir() else None, lstm_rmse  if 'lstm_rmse'  in dir() else None],
    'MAPE_%': [arima_mape,  sarima_mape,  prophet_mape  if 'prophet_mape'  in dir() else None, lstm_mape  if 'lstm_mape'  in dir() else None],
}
eval_df = pd.DataFrame(eval_data).dropna()
eval_df = eval_df.sort_values('MAPE_%').reset_index(drop=True)
eval_df['Rank'] = eval_df.index + 1

print(eval_df.to_string(index=False))
eval_df.to_csv('outputs/powerbi_model_evaluation.csv', index=False)
print("\n✓ Saved: outputs/powerbi_model_evaluation.csv")

# ── Model comparison bar chart ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Model Evaluation — Bitcoin Price Forecasting', fontsize=14, color=ACCENT, fontweight='bold')

bar_colors = [ACCENT, ACCENT2, ACCENT3, ACCENT4][:len(eval_df)]
for ax, metric in zip(axes, ['MAE', 'RMSE', 'MAPE_%']):
    bars = ax.bar(eval_df['Model'], eval_df[metric], color=bar_colors, alpha=0.85, width=0.5)
    ax.set_title(metric, fontsize=11, color='white')
    ax.set_xticklabels(eval_df['Model'], rotation=20, ha='right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, eval_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f'{val:,.0f}' if metric != 'MAPE_%' else f'{val:.1f}%',
                ha='center', va='bottom', fontsize=8, color='white')

plt.tight_layout()
plt.savefig('outputs/13_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot 13 saved: Model comparison")


# =============================================================================
# SECTION 8 — INTERACTIVE PLOTLY CHARTS (for Streamlit/HTML export)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 8 — INTERACTIVE PLOTLY CHARTS")
print("=" * 60)

# ── Candlestick chart ─────────────────────────────────────────────────────────
btc_candle = df[df['Name']=='Bitcoin'].sort_values('Date').tail(365)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.03, row_heights=[0.75, 0.25])

fig.add_trace(go.Candlestick(
    x=btc_candle['Date'], open=btc_candle['Open'], high=btc_candle['High'],
    low=btc_candle['Low'], close=btc_candle['Close'],
    name='BTC', increasing_line_color=ACCENT3, decreasing_line_color=ACCENT), row=1, col=1)

fig.add_trace(go.Scatter(x=btc_candle['Date'], y=btc_candle['MA_30'],
    line=dict(color=ACCENT2, width=1.5), name='MA 30'), row=1, col=1)
fig.add_trace(go.Scatter(x=btc_candle['Date'], y=btc_candle['BB_Upper'],
    line=dict(color='rgba(121,192,255,0.4)', width=0.8), name='BB Upper'), row=1, col=1)
fig.add_trace(go.Scatter(x=btc_candle['Date'], y=btc_candle['BB_Lower'],
    line=dict(color='rgba(121,192,255,0.4)', width=0.8), name='BB Lower',
    fill='tonexty', fillcolor='rgba(121,192,255,0.05)'), row=1, col=1)

fig.add_trace(go.Bar(x=btc_candle['Date'], y=btc_candle['Volume'],
    marker_color=ACCENT2, opacity=0.5, name='Volume'), row=2, col=1)

fig.update_layout(
    title='Bitcoin — Interactive Candlestick Chart (Last 365 Days)',
    xaxis_rangeslider_visible=False,
    template='plotly_dark',
    height=650,
    legend=dict(orientation='h', y=1.02)
)
fig.write_html('outputs/interactive_candlestick.html')
fig.show()
print("✓ Saved: outputs/interactive_candlestick.html")

# ── Interactive multi-coin normalized price ───────────────────────────────────
fig2 = go.Figure()
coins_plot = ['Bitcoin','Ethereum','Dogecoin','XRP','Litecoin','Solana','Cardano']
for coin in coins_plot:
    sub = df[df['Name']==coin].sort_values('Date')
    sub = sub[sub['Close']>0]
    norm = sub['Close'] / sub['Close'].iloc[0]
    fig2.add_trace(go.Scatter(x=sub['Date'], y=norm, name=coin, mode='lines'))

fig2.update_layout(
    title='Normalized Price — Multi-coin Comparison',
    yaxis_title='Normalized Price (Base = 1)',
    template='plotly_dark', height=500,
    hovermode='x unified'
)
fig2.write_html('outputs/interactive_multicoin.html')
fig2.show()
print("✓ Saved: outputs/interactive_multicoin.html")

# ── Volatility heatmap by year/month (BTC) ────────────────────────────────────
btc_vol = df[df['Name']=='Bitcoin'][['Date','Volatility_30']].dropna().copy()
btc_vol['Year']  = btc_vol['Date'].dt.year
btc_vol['Month'] = btc_vol['Date'].dt.month
vol_pivot = btc_vol.groupby(['Year','Month'])['Volatility_30'].mean().unstack()

fig3 = go.Figure(go.Heatmap(
    z=vol_pivot.values,
    x=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
    y=vol_pivot.index.tolist(),
    colorscale='RdYlGn_r',
    text=np.round(vol_pivot.values, 2),
    texttemplate='%{text}',
    showscale=True
))
fig3.update_layout(
    title='Bitcoin — Average Monthly Volatility Heatmap (by Year)',
    template='plotly_dark', height=400
)
fig3.write_html('outputs/interactive_vol_heatmap.html')
fig3.show()
print("✓ Saved: outputs/interactive_vol_heatmap.html")


# =============================================================================
# SECTION 9 — POWER BI EXPORT SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 9 — POWER BI EXPORT FILES")
print("=" * 60)

powerbi_files = [
    ("featured_crypto.csv",        "Master dataset — all coins, all features. Main slicer source."),
    ("powerbi_correlation.csv",    "Correlation matrix → Page 7 heatmap visual"),
    ("powerbi_arima_sarima.csv",   "ARIMA & SARIMA predictions → Page 3 & 9"),
    ("powerbi_prophet.csv",        "Prophet forecast + uncertainty bands → Page 3"),
    ("powerbi_lstm.csv",           "LSTM predictions → Page 3 & 9"),
    ("powerbi_model_evaluation.csv","MAE/RMSE/MAPE table → Page 9 leaderboard"),
    ("stationarity_tests.csv",     "ADF test results → Page 8 explainability"),
]

print(f"\n{'File':<40} {'Usage'}")
print("-" * 80)
for fname, usage in powerbi_files:
    exists = "✓" if os.path.exists(f"outputs/{fname}") else "✗"
    print(f"  {exists} {fname:<38} {usage}")

print("\n" + "=" * 60)
print("ALL DONE! Check the outputs/ folder.")
print("=" * 60)
print("""
NEXT STEPS:
  1. Open Power BI Desktop
  2. Home → Get Data → Text/CSV
  3. Import: featured_crypto.csv  (main)
  4. Import all powerbi_*.csv files
  5. Create relationships on [Date] and [Name/Symbol]
  6. Build your 10 dashboard pages using these tables
""")
