import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from attention import Attention, FeatureAttention
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dropout, Dense, GRU, Bidirectional, Input, Multiply, Lambda, multiply, MultiHeadAttention, LayerNormalization, Conv1D,Input, MultiHeadAttention, Dense, LayerNormalization, Dropout, Add, GlobalAveragePooling1D
from time_series import *
from keras.layers.core import *
from keras import backend as K
from ta.trend import macd, cci, sma_indicator
from ta.momentum import roc, rsi
from ta.volatility import bollinger_wband
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
import traceback
from tensorflow.keras.layers import Input, Conv1D, SeparableConv1D, BatchNormalization, ReLU, Add, GlobalAveragePooling1D, Dropout
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Add,SeparableConv1D
from tensorflow.keras.layers import MultiHeadAttention, Dense, LayerNormalization, Add, GlobalAveragePooling1D, Dropout
from tensorflow.keras.models import Model
import time


np.random.seed(seed)
tf.random.set_seed(seed)

# load dataset
# if interval=="Hourly":
#
#     df = pd.read_csv(full_time_series_path, parse_dates=[['Date', 'Time']])
# if interval=="Daily":
# Define constants
log_file = "training_log.txt"  # File to track progress
completed_tasks = set()  # Keeps track of completed tasks
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        completed_tasks = set(line.strip() for line in f)

np.random.seed(seed)
tf.random.set_seed(seed)

# Load dataset





def transformer_block(inputs, num_heads=4, key_dim=64, ff_dim=128, dropout_rate=0.1):
    """A deeper Transformer block with residual connections"""
    # Self-attention layer
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)

    # Residual connection and normalization
    x = Add()([inputs, attn_output])
    x = LayerNormalization()(x)

    # Feedforward network
    ff_output = Dense(ff_dim, activation="relu")(x)
    ff_output = Dropout(dropout_rate)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)

    # Residual connection and normalization
    x = Add()([x, ff_output])
    x = LayerNormalization()(x)

    return x


def DeepTransformer(inp, num_layers=4, num_heads=4, key_dim=64, ff_dim=128, dropout_rate=0.1):
    """Transformer model with increased depth"""
    x = inp

    # Stack multiple Transformer blocks
    for _ in range(num_layers):
        x = transformer_block(x, num_heads=num_heads, key_dim=key_dim, ff_dim=ff_dim, dropout_rate=dropout_rate)

    # Global Pooling for feature compression
    x = GlobalAveragePooling1D()(x)

    # Final Dense layers
    x = Dense(256, activation="relu")(x)

    return x


features = [ "High","Low","Open","Close","High_Open","High_Low", "High_Close", \
             "Low_Open", "Low_Close", "Open_Close",\
             "High_Low_Open_Close",\
            "SMA_MACD_ROC_RSI_BB_CCI"]

features = ["Low_Close"]
# model_name = ["LSTM", "Attention", "Bidirectional", "GRU", "stack_lstm", "TCN", "Transformer"]
model_name = [ "Attention"]
tickers = ["GBPJPY","USDJPY","EURUSD"]

early_stopping = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)


# Function for training a model
def train_model(ticker,feature, md_name, w):
    try:
        model_path = f'models/{ticker}/{interval}/{Problem}/{md_name}/{feature}-{w}.h5'
        print(model_path)
        if os.path.exists(model_path):
            print(f"Model already exists: {model_path}. Skipping...")
            return True
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

        X_train, y_train, X_val, y_val, X_test, y_test = get_data(ticker,feature, df, w)
        inp = Input(shape=(X_train.shape[1], X_train.shape[2]))

        # Define the model architecture
        if md_name == "Bidirectional":
            layer1 = Bidirectional(LSTM(76, batch_size=batch_size, return_sequences=False))(inp)
        elif md_name == "LSTM":
            layer1 = LSTM(76, batch_size=batch_size, return_sequences=False)(inp)
        elif md_name == "GRU":
            layer1 = GRU(76, batch_size=batch_size, return_sequences=False)(inp)
        elif md_name == "Attention":
            lstm_output = LSTM(64, return_sequences=True, name="LSTM")(inp)
            attention_layer = FeatureAttention(num_features=X_train.shape[2], hidden_units=128, name="FeatureAttention")
            layer1, attention_weights = attention_layer(lstm_output)

            # layer0 = LSTM(76, batch_size=batch_size, return_sequences=True)(inp)
            # layer1 = Attention(units=100)(layer0)
        elif md_name == "stack_lstm":
            layer0 = LSTM(76, batch_size=batch_size, return_sequences=True)(inp)
            layer1 = LSTM(100, batch_size=batch_size, return_sequences=True)(layer0)
        elif md_name == "TCN":
            # First TCN block
            tcn_layer = Conv1D(filters=256, kernel_size=3, dilation_rate=1, padding="causal", activation="relu")(inp)
            tcn_layer = Conv1D(filters=76, kernel_size=3, dilation_rate=4, padding="causal", activation="relu")(
                tcn_layer)
            layer1 = GlobalAveragePooling1D()(tcn_layer)
        elif md_name == "Transformer":
            x = DeepTransformer(inp, num_layers=4, num_heads=4, key_dim=256, ff_dim=256, dropout_rate=0.1)

            layer1 = Dense(76, activation="relu")(x)
        else:
            print(f"Unknown model type: {md_name}")
            return False

        dense_1 = Dense(1, activation="linear", name="Output")(layer1)
        # print(dense_2.shape)
        # optimizer = Adam(clipvalue=0.5)

        adam = tf.keras.optimizers.Adam(clipvalue=0.5, learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=None,
                                        decay=0.001, amsgrad=False)
        model = Model(inputs=inp, outputs=dense_1)
        model.compile(loss="mean_squared_error", optimizer=adam)

        # print(model.output_names)
        # exit()
        # model.compile(optimizer="adam", loss={"Output": "mse"}, loss_weights={"Output": 1.0})
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                            shuffle=False, verbose=2, callbacks=[early_stopping])
        model.save(model_path)
        # print(model.summary())
        # exit()
        print(f"Model saved: {model_path}")

        K.clear_session()
        tf.compat.v1.reset_default_graph()
        return True
    except Exception as e:
        print(f"Error during training for {feature}, {md_name}, window {w}: {e}")
        traceback.print_exc()
        return False


# Monitor and retry training
for ticker in tickers:
    for feature in features:
        for md_name in model_name:
            if ticker=="EURUSD":
                window_size = [16]
                epochs=30
            elif ticker=="GBPJPY":
                window_size = [16]
                epochs=30
            else:
                window_size = [16]
                epochs=20


            for w in window_size:
                task_id = f"{ticker}_{feature}_{md_name}_{w}"
                if task_id in completed_tasks:
                    print(f"Task already completed: {task_id}. Skipping...")
                    continue

                success = False
                while not success:
                    success = train_model(ticker,feature, md_name, w)
                    if not success:
                        print(f"Retrying task: {task_id} after failure...")
                        time.sleep(5)  # Pause before retrying

                # Log completed task
                with open(log_file, "a") as log:
                    log.write(f"{task_id}\n")
                    completed_tasks.add(task_id)
