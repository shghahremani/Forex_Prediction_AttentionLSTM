import os
import time
import traceback
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import norm
from ta.trend import macd, cci, sma_indicator
from ta.momentum import roc, rsi
from ta.volatility import bollinger_wband
from common_variables import *
from attention import Attention, FeatureAttention
from time_series import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


log_file = "evaluating_log.txt"  # File to track progress
completed_tasks = set()  # Keeps track of completed tasks
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        completed_tasks = set(line.strip() for line in f)

np.random.seed(seed)
tf.random.set_seed(seed)

def cohen_d(x, y):
    diff = np.mean(x) - np.mean(y)
    pooled_std = np.sqrt((np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2)
    return diff / pooled_std

def calculate_confidence_interval(errors):
    mae = np.mean(errors)
    se_mae = np.std(errors) / np.sqrt(len(errors))
    z = norm.ppf(0.975)
    ci_lower_normal = mae - z * se_mae
    ci_upper_normal = mae + z * se_mae
    CI_interval = (ci_upper_normal - ci_lower_normal) / 2
    return CI_interval

features = ["High", "Low", "Open", "Close", "High_Low", "High_Open", "High_Close",
            "Low_Open", "Low_Close", "Open_Close", "High_Low_Open_Close", "SMA_MACD_ROC_RSI_BB_CCI"]

window_size = [2,4,8]
# model_name = ["LSTM", "Attention", "Bidirectional", "GRU", "stack_lstm", "TCN", "Transformer"]
model_name = ["LSTM"]

tickers = ["GBPJPY"]

def evaluate_model(ticker, feature, md_name, window_size):
    try:
        csv_filename = f'results/{ticker}/{md_name}/{feature}_best_results.csv'
        print(csv_filename)
        if os.path.exists(csv_filename):
            print(f"Results already exist: {csv_filename}. Skipping...")
            return True

        scaler_path = f'scalers/{ticker}/{interval}/{Problem}/label.bin'
        scaler = joblib.load(scaler_path)
        full_time_series_path = f'data/{interval}/{ticker}.csv'
        df = pd.read_csv(full_time_series_path)

        df['SMA'] = sma_indicator(close=df['Close'], window=10, fillna=False)
        df['MACD'] = macd(close=df['Close'], window_slow=12, window_fast=26, fillna=False)
        df['ROC'] = roc(close=df['Close'], window=2, fillna=False)
        df['RSI'] = rsi(close=df['Close'], window=10, fillna=False)
        df['BB'] = bollinger_wband(close=df['Close'], window=20, fillna=False)
        df['CCI'] = cci(high=df['High'], low=df['Low'], close=df['Close'], window=20, fillna=False)
        df.dropna(how='any', inplace=True)
        df = df[df.shape[0] % batch_size:]

        best_mae = float('inf')
        best_result = {}

        for w in window_size:
            X_train, y_train, X_val, y_val, X_test, y_true = get_data(ticker, feature, df, w)
            # print(np.array(X_test.shape))
            # print(np.array(y_true).shape)
            # exit()
            model_path = f'models/{ticker}/{interval}/{Problem}/{md_name}/{feature}-{w}.h5'

            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}. Skipping...")
                continue

            model = load_model(model_path, custom_objects={'FeatureAttention': FeatureAttention})
            start_time = time.time()
            y_pred = model.predict(X_test)
            end_time = time.time()

            inference_time = (end_time - start_time) / len(X_test)
            if md_name=="stack_lstm":
                y_pred=np.mean(y_pred, axis=1)  # Shape becomes (batch_size, 1)

            y_pred = scaler.inverse_transform(y_pred)
            y_true = scaler.inverse_transform(y_true)

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_true, y_pred)

            mae_ci = calculate_confidence_interval(np.abs(y_true - y_pred))
            rmse_ci = calculate_confidence_interval((y_true - y_pred) ** 2)
            mape_ci = calculate_confidence_interval(np.abs((y_true - y_pred) / y_true))

            if mae < best_mae:
                best_mae = mae
                best_result = {
                    "Feature": feature,
                    "Ticker": ticker,
                    "Model": md_name,
                    "Best Window Size": w,
                    "MAE": mae,
                    "MAE CI": mae_ci,
                    "RMSE": rmse,
                    "RMSE CI": rmse_ci,
                    "MAPE": mape,
                    "MAPE CI": mape_ci,
                    "Inference Time (s)": inference_time
                }

        results_df = pd.DataFrame([best_result])
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        results_df.to_csv(csv_filename, index=False)

        print(f"Best results saved to {csv_filename}")
        K.clear_session()
        return True

    except Exception as e:
        print(f"Error during evaluation for {feature}, {md_name}, window {locals().get('w', 'N/A')}: {e}")
        traceback.print_exc()
        return False

for ticker in tickers:
    for feature in features:
        for md_name in model_name:
            task_id = f"{ticker}_{feature}_{md_name}"
            if task_id in completed_tasks:
                print(f"Task already completed: {task_id}. Skipping...")
                continue

            success = False
            while not success:
                success = evaluate_model(ticker, feature, md_name, window_size)
                if not success:
                    print(f"Retrying task: {task_id} after failure...")
                    time.sleep(5)

            with open(log_file, "a") as log:
                log.write(f"{task_id}\n")
                completed_tasks.add(task_id)

    # Merge all results for each model and each ticker
    for md_name in model_name:
        for ticker in tickers:
            result_files = [os.path.join(root, file) for root, _, files in os.walk(f'results/{ticker}/{md_name}') for
                            file in files if file.endswith('_best_results.csv')]
            if result_files:
                merged_results = pd.concat([pd.read_csv(file) for file in result_files], ignore_index=True)
                os.makedirs(f'results/{ticker}/{md_name}', exist_ok=True)
                merged_results.to_csv(f'results/{ticker}/{md_name}/all_results.csv', index=False)
                print(f"All results merged and saved to results/{ticker}/{md_name}/all_results.csv")

# total_gain = 0
# current_price=y_true[0]
# # We assume the position is closed each time after one prediction
# for i in range(1,len(y_true) - 1):
#     if y_pred[i] > current_price:
#         # Predicted increase, buy at y_true[i] and sell at y_true[i+1]
#         gain = (y_true[i] - current_price)*1000
#
#         # print("current price",current_price)
#         # print("prediction",y_pred[i])
#         # print("true_value",y_true[i])
#         # print("profit",gain)
#         current_price = y_true[i]
#     elif y_pred[i] < current_price:
#         # Predicted decrease, sell at y_true[i] and buy back at y_true[i+1]
#         gain = (current_price - y_true[i])*1000
#         current_price = y_true[i]
#         # print("current price", current_price)
#         # print("prediction", y_pred[i])
#         # print("true_value", y_true[i])
#         # print("profit", gain)
#         current_price = y_true[i]
#
#     else:
#         # No trade if predictions equal the current price
#         gain = 0
#     total_gain += gain
#
# print(f'Total Gain: {total_gain}')
#
