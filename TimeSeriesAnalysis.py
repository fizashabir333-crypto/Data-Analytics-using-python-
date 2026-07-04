
# Time Series Analysis


# Import libraries

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA


# Load dataset


df = pd.read_csv("airline-passengers.csv")

print(df.head())
print(df.columns)

# Convert Month column


# ==========================================
# Convert month column
# ==========================================

df["month"] = pd.to_datetime(
    df["month"]
)

# Set month as index
df.set_index(
    "month",
    inplace=True
)

# Target variable
series = df["total_passengers"]

print(series.head())

# # Original time series plot


plt.figure(figsize=(10,6))

plt.plot(
    series
)

plt.title(
    "Monthly Airline Passengers"
)

plt.xlabel(
    "Year"
)

plt.ylabel(
    "Passengers"
)

plt.show()


# Time Series Decomposition


decomposition = seasonal_decompose(
    series,
    model='additive',
    period=12
)

decomposition.plot()

plt.show()


# Moving Average


moving_avg = series.rolling(
    window=12
).mean()

plt.figure(figsize=(10,6))

plt.plot(
    series,
    label="Original"
)

plt.plot(
    moving_avg,
    label="Moving Average"
)

plt.legend()

plt.title(
    "Moving Average Analysis"
)

plt.show()


# ARIMA Forecasting


model = ARIMA(
    series,
    order=(1,1,1)
)

model_fit = model.fit()

forecast = model_fit.forecast(
    steps=12
)

print(
    "\nForecasted Values:"
)

print(
    forecast
)


# Forecast Visualization


plt.figure(figsize=(10,6))

plt.plot(
    series,
    label="Original Data"
)

plt.plot(
    forecast,
    label="Forecast"
)

plt.legend()

plt.title(
    "ARIMA Forecast"
)

plt.show()