# MIT License
# Copyright (c) 2020 Adam Tibi (https://linkedin.com/in/adamtibi/ , https://adamtibi.net)
# ticker = 'USDJPY' # Your data file name without extention
interval="Hourly" #Hourly or Daily
Problem="Regression" # Regression or Catagorical
if interval=="Hourly":
    val_size=300
elif interval=="Daily":
    val_size = 2

# full_time_series_path = f'data/{interval}/{ticker}.csv'
batch_size = 32
validation_size = val_size * batch_size # must be a multiple of batch_size
test_size = val_size * batch_size # must be a multiple of batch_size
# ma_periods = 5 # Simple Moving Average periods length
# start_date = '2014-03-19' # Ignore any data in the file prior to this date
seed = 52 # An arbitrary value to make sure your seed is the same


