import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from attention import Attention
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dropout, Dense, GRU, Bidirectional, Input, Multiply, Lambda, multiply, MultiHeadAttention, LayerNormalization, Conv1D
from time_series import *
from keras.layers.core import *
from keras import backend as K
from ta.trend import macd, cci, sma_indicator
from ta.momentum import roc, rsi
from ta.volatility import bollinger_wband
from keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint

# In[ ]:


epochs = 800
np.random.seed(seed)
tf.random.set_seed(seed)
interval="Daily"
# load dataset
if interval=="Hourly":
    df = pd.read_csv(full_time_series_path, parse_dates=[['Date', 'Time']])
if interval=="Daily":
    df = pd.read_csv(full_time_series_path)
df['SMA'] = sma_indicator(close=df['Close'], window=10, fillna=False)
df['MACD'] = macd(close=df['Close'], window_slow=12, window_fast=26, fillna=False)
df['ROC'] = roc(close=df['Close'], window=2, fillna=False)
df['RSI'] = rsi(close=df['Close'], window=10, fillna=False)
df['BB'] = bollinger_wband(close=df['Close'], window=20, fillna=False)
df['CCI'] = cci(high=df['High'], low=df['Low'], close=df['Close'], window=20, fillna=False)
df.dropna(how='any', inplace=True)
df = df[df.shape[0] % batch_size:]

features = [ "High_Open", "High_Close", \
             "Low_Open", "Low_Close", "Open_Close",\
             "High_Low_Open_Close",\
            "SMA_MACD_ROC_RSI_BB_CCI"]
# features = [ "High_Close"]
# model_name = ["LSTM", "Attention","Bidirectional","GRU"]
model_name = ["LSTM","Attention",]

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

for feature in (features):
    for md_name in model_name:
        # window_size = [ 2]
        window_size = [14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

        for w in window_size:
            X_train, y_train, X_val, y_val, X_test, y_test = get_data(feature, df, w)
            #exit()


            inp = Input(shape=(X_train.shape[1], X_train.shape[2]))
            # print(X.shape[1])

            if md_name == "Bidirectional":
                layer1 = Bidirectional(LSTM(76, batch_size=batch_size, return_sequences=False))(inp)
            elif md_name == "LSTM":
                layer1 = LSTM(76, batch_size=batch_size, return_sequences=False)(inp)
            elif md_name == "Attention":

                layer0 = LSTM(76, batch_size=batch_size, return_sequences=True)(inp)
                layer1 = Attention(units=100 )(layer0)

            elif md_name == "Attention_stacked":
                layer0 = LSTM(152, batch_size=batch_size, return_sequences=True)(inp)
                layer00 = LSTM(76, batch_size=batch_size, return_sequences=True)(layer0)
                # print(layer0.shape)
                layer1 = Attention(units=100 )(layer0)
            elif md_name == "Attention_input":
                layer0 = Attention(inp)
                layer1 = LSTM(76, batch_size=batch_size, return_sequences=False)(layer0)
                print(layer1.shape)
                # layer1 = attention_3d_block(layer1, w, 'attention_input')
            elif md_name == "GRU":
                layer1 = GRU(76, batch_size=batch_size, return_sequences=False)(inp)

            # dropout_1 = Dropout(0.2)(layer1)
            dense_1 = Dense(units=1)(layer1)
            # print(dense_2.shape)
            # optimizer = Adam(clipvalue=0.5)
            adam = tf.keras.optimizers.Adam(clipvalue=0.5, learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=None,
                                            decay=0.001, amsgrad=False)
            model = Model(inputs=inp, outputs=dense_1)
            model.compile(loss="mean_squared_error", optimizer=adam)

            # print(model.summary())

            # In[ ]:
            # Define a ModelCheckpoint callback
            checkpoint = ModelCheckpoint(
                filepath='best_model.h5',
                monitor='val_loss',  # Metric to monitor for saving the best model
                save_best_only=True,  # Save only the best model
                save_weights_only=False,  # Save the entire model (including architecture)
                mode='min',
                # 'min' if the monitored metric should be minimized (e.g., val_loss), 'max' if it should be maximized (e.g., accuracy)
                verbose=1
            )

            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                                shuffle=False, verbose=2,callbacks=[early_stopping])

            # # In[ ]:
            # if md_name == "LSTM":
            #     model_name_dir = "LSTM"
            # elif md_name == "Bidirectional:
            #     model_name_dir = "Bidirectional"
            # elif md_name == "Attention":
            #     model_name_dir = "Attention"
            # elif md_name == "Attention_input":
            #     model_name_dir = "Attention_input"
            # elif md_name == "Attention_stacked":
            #     model_name_dir = "Attention_stacked"
            # else:
            #     model_name_dir = "GRU"
            model_path = f'models/{interval}/{md_name}/{feature}-{w}.h5'
            print(model_path)
            model.save(model_path)

# In[ ]:


fig = plt.figure(figsize=(12, 8))
ax1 = fig.subplots(1)
ax1.set_title('Model Loss')
ax1.set(xlabel='Epoch', ylabel='Loss')
ax1.plot(history.history['loss'][7:], label='Train Loss')
ax1.plot(history.history['val_loss'][7:], label='Val Loss')
ax1.legend()

plt.show()
# In[ ]:
