import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import warnings

import os

warnings.filterwarnings('ignore')

# 1. Dataset Acquisition

full_df = pd.read_csv('./all_stocks_5yr.csv')

# Filter for a single stock (Apple - AAPL)
df = full_df[full_df['Name'] == 'AAPL'].copy()

# Rename columns to standard capitalization expected by downstream code
df = df.rename(columns={
    'date': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
print("Dataset successfully downloaded and loaded!")

# 2. Dataset Exploration and Datetime Formatting
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# 3. Historical Stock Price Trends
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], color='royalblue')
plt.title('Historical Closing Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()

# 4. Moving Averages
df['MA30'] = df['Close'].rolling(window=30).mean()
df['MA100'] = df['Close'].rolling(window=100).mean()

plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Original Close', alpha=0.5)
plt.plot(df['MA30'], label='30-Day Moving Average', color='orange')
plt.plot(df['MA100'], label='100-Day Moving Average', color='crimson')
plt.title('Stock Prices with Rolling Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# 5. Seasonality Analysis
# Resample to weekly to reduce noise in decomposition
weekly_close = df['Close'].resample('W').mean().ffill()
decomposition = seasonal_decompose(weekly_close, model='additive', period=52)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

# 6. Correlation Analysis
correlation_matrix = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 7. ARIMA Forecasting
# Train the model on weekly averaged data for smoother forecasting
model = ARIMA(weekly_close, order=(5, 1, 0))
results = model.fit()

forecast_steps = 26
forecast = results.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=weekly_close.index[-1], periods=forecast_steps + 1, freq='W')[1:]
forecast_series = pd.Series(forecast.predicted_mean.values, index=forecast_index)

plt.figure(figsize=(12, 6))
plt.plot(weekly_close[-104:], label='Historical Weekly Close')
plt.plot(forecast_series, label='Forecasted Price', color='red', linestyle='--')
plt.title('Stock Price Forecasting (Next 6 Months)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()
