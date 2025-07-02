import requests
import pandas as pd
from datetime import datetime
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import MetaTrader5 as mt5  # Make sure you have the MetaTrader5
# Alpha Vantage API parameters
# MetaTrader account credentials
account = 5032862937  # Replace with your MetaTrader account number
password = "@xKm8fZw"  # Replace with your MetaTrader account password
server = "MetaQuotes-Demo"  # Replace with your MetaTrader server name

# Connect to the MetaTrader terminal
# Connect to the MetaTrader terminal
if not mt5.initialize(login=account, password=password, server=server):
    print("initialize() failed, error code:", mt5.last_error())
    mt5.shutdown()
    quit()

# Set the necessary parameters
symbol = "EURUSD"
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    print(f"Symbol {symbol} not found")
    mt5.shutdown()
    quit()

# Set the timeframe to daily (or hourly as needed)
timeframe = mt5.TIMEFRAME_H1  # Daily timeframe, use mt5.TIMEFRAME_H1 for hourly

# Define the date range from 2014 to 2024
from_date = datetime(2024,6, 8)
to_date = datetime(2024, 9, 8)

# Download historical data for the specified date range
rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)

# Convert the rates to a structured NumPy array
rates = np.rec.array(rates,
                     dtype=[('time', '<M8[s]'), ('open', '<f8'), ('high', '<f8'), ('low', '<f8'),
                            ('close', '<f8'), ('tick_volume', '<i8'), ('spread', '<i4'), ('real_volume', '<i8')])

# Prepare the data for prediction
time = rates.time
close_prices = rates.close
high_prices = rates.high
low_prices = rates.low
volume = rates.tick_volume
open_prices = rates.open
print(open_prices)
# Create a DataFrame
data = {
    'Time': time,
    'Open': open_prices,
    'High': high_prices,
    'Low': low_prices,
    'Close': close_prices,
    'Volume': volume
}
df = pd.DataFrame(data)

# Convert time to a human-readable format
df['Time'] = pd.to_datetime(df['Time'], unit='s')  # Convert UNIX timestamp to datetime

# Save DataFrame to CSV

csv_filename = f'data/hourly/{symbol}_RECENT.csv'
df.to_csv(csv_filename, index=False)

# Shut down the connection to MetaTrader 5
mt5.shutdown()