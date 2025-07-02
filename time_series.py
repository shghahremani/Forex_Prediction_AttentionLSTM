# MIT License
# Copyright (c) 2020 Adam Tibi (https://linkedin.com/in/adamtibi/ , https://adamtibi.net)
from datetime import datetime
import numpy as np
from common_variables import *
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.utils import to_categorical
import os
def get_data_test(ticker,colomn,df, window_size):
    print("First element:", df["Time"].iat[0])
    print("Last element:", df["Time"].iat[-1])

    df_y = df[["Close"]]
    if colomn == "High":
        df = df[["High"]]
    elif colomn == "Low":
        df = df[["Low"]]
    elif colomn == "Volume":
        df = df[["Volume"]]
    elif colomn == "Open":
        df = df[["Open"]]
    elif colomn == "Close":
        df = df[["Close"]]
    elif colomn == "High_Low":
        df = df[["High", "Low"]]
    elif colomn == "High_Volume":
        df = df[["High", "Volume"]]
    elif colomn == "High_Open":
        df = df[["High", "Open"]]
    elif colomn == "High_Close":
        df = df[["High", "Close"]]
    elif colomn == "Low_Volume":
        df = df[["Low", "Volume"]]
    elif colomn == "Low_Open":
        df = df[["Low", "Open"]]
    elif colomn == "Low_Close":
        # time = df[['Time']]
        df = df[["Low", "Close"]]



    elif colomn == "Volume_Open":
        df = df[["Volume", "Open"]]
    elif colomn == "Volume_Close":
        df = df[["Volume", "Close"]]
    elif colomn == "Open_Close":
        df = df[["Open", "Close"]]
    elif colomn == "High_Low_Volume_Open_Close":
        df = df[["High", "Low", "Volume", "Open", "Close"]]

    elif colomn == "High_Low_Open_Close":

        df = df[["High", "Low", "Open", "Close"]]
        df
    elif colomn == "High_Low_Close":

        df = df[["High", "Low", "Close"]]
        # print(df.values)
    elif colomn == "SMA_MACD_ROC_RSI_BB_CCI":

        df = df[["SMA", "MACD", "ROC", "RSI", "BB", "CCI"]]


    df_test = df
    # time=time[173:].values
    # print(df_test)
    # exit()
    # df_test.to_csv('data/Hourly/catagorical/test_EURUSD.csv', index=False)
    # exit()
    # train_values = df_train.values

    scaler_path = f'scalers/{ticker}/{interval}/{Problem}/{colomn}.bin'
    scaler = joblib.load(scaler_path)
    test_values = scaler.transform(df_test.values)
    # test_values = df_test.values

    # print(np.array(df_test.values).shape)
    # exit()

    df_test_y = df_y
    # print(df_test_y)
    # exit()

    scaler_path = f'scalers/{ticker}/{interval}/{Problem}/label.bin'
    scaler_y = joblib.load(scaler_path)

    # print(df_test_y.shape)
    test_values_y = scaler_y.transform(df_test_y.values)
    # print(test_values_y)
    # exit()


    X_test, y_test = [], []


    len_values = len(test_values)
    # print(np.array(test_values).shape)
    # print(test_values)
    for i in range(window_size, len_values):
        X_test.append(test_values[i - window_size:i])
        y_test.append(test_values_y[i])
        # X_time.append(time[i])

    X_test, y_test = np.asarray(X_test), np.asarray(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    return  X_test, y_test,scaler,scaler_y

def get_data(ticker,colomn,df, window_size):
    a=["High", "Low", "Volume", "Open", "Close", "High_Low", "High_Volume", "High_Open", "High_Close", \
     "Low_Volume", "Low_Open", "Low_Close", "Volume_Open", "Volume_Close", "Open_Close", "High_Low_Volume_Open_Close", \
     "SMA_MACD_ROC_RSI_BB_CCI"]

    df_y=df[["Close"]]
    if colomn=="High":
        df=df[["High"]]
    elif colomn=="Low":
        df=df[["Low"]]
    elif colomn=="Volume":
        df=df[["Volume"]]
    elif colomn == "Open":
        df = df[["Open"]]
    elif colomn == "Close":
        df = df[["Close"]]
    elif colomn == "High_Low":
        df = df[["High","Low"]]
    elif colomn == "High_Volume":
        df = df[["High", "Volume"]]
    elif colomn == "High_Open":
        df = df[["High", "Open"]]
    elif colomn == "High_Close":
        df = df[["High", "Close"]]
    elif colomn == "Low_Volume":
        df = df[["Low", "Volume"]]
    elif colomn == "Low_Open":
        df = df[["Low", "Open"]]
    elif colomn == "Low_Close":
        # time = df[['Time']]
        df = df[["Low", "Close"]]


        # time = df[['Time']]

    elif colomn == "Volume_Open":
        df = df[["Volume", "Open"]]
    elif colomn == "Volume_Close":
        df = df[["Volume", "Close"]]
    elif colomn == "Open_Close":
        df = df[["Open", "Close"]]
    elif colomn == "High_Low_Volume_Open_Close":
        df = df[["High", "Low", "Volume", "Open", "Close"]]

    elif colomn == "High_Low_Open_Close":

        df = df[["High", "Low", "Open", "Close"]]
        df
    elif colomn == "High_Low_Close":

        df = df[["High", "Low", "Close"]]
        # print(df.values)
    elif colomn == "SMA_MACD_ROC_RSI_BB_CCI":

        df = df[["SMA", "MACD", "ROC", "RSI", "BB", "CCI"]]

    validation_size=int(0.2*len(df))
    test_size=int(0.2*len(df))
    # print(validation_size)
    # print(len(df))
    df_train = df[:- validation_size - test_size]
    # print(df_train.values)
    # exit()
    # df_second_dataset=df[- (validation_size+test_size):]
    # csv_filename = 'historical_data_second_model.csv'
    # df_second_dataset.to_csv(csv_filename, index=False)
    df_validation = df[- validation_size - test_size:- test_size]
    df_test = df[-test_size:]
    # time=time[173:].values
    # print(df_test)
    # exit()
    # df_test.to_csv('data/Hourly/catagorical/test_EURUSD.csv', index=False)
    # exit()
    scaler = MinMaxScaler()
    train_values = scaler.fit_transform(df_train.values)
    # train_values = df_train.values

    scaler_path = f'scalers/{ticker}/{interval}/{Problem}/{colomn}.bin'
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    joblib.dump(scaler, scaler_path)
    val_values = scaler.transform(df_validation.values)
    test_values = scaler.transform(df_test.values)
    # test_values = df_test.values

    # print(np.array(df_test.values).shape)
    # exit()

    df_train_y = df_y[:- validation_size - test_size]
    df_validation_y = df_y[- validation_size - test_size:- test_size]
    df_test_y = df_y[-test_size:]
    # print(df_test_y)
    # exit()
    scaler_y = MinMaxScaler()
    train_values_y = scaler_y.fit_transform(df_train_y.values)
    scaler_path = f'scalers/{ticker}/{interval}/{Problem}/label.bin'
    joblib.dump(scaler_y, scaler_path)
    val_values_y = scaler_y.transform(df_validation_y.values)
    # print(df_test_y.shape)
    test_values_y = scaler_y.transform(df_test_y.values)
    # print(test_values_y)
    # exit()

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    X_time=[]
    #train_values=np.array(train_values).squeeze()
    len_values = len(train_values)
    for i in range(window_size, len_values):
        X_train.append(train_values[i-window_size:i,:])
        y_train.append(train_values_y[i])
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    #print(f"X {X_train.shape}, y {y_train.shape}")

    len_values = len(val_values)
    for i in range(window_size, len_values):
        X_val.append(val_values[i-window_size:i,:])
        y_val.append(val_values_y[i])
    X_val, y_val = np.asarray(X_val), np.asarray(y_val)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2]))
    #print(f"X {X_val.shape}, y {y_val.shape}")

    len_values = len(test_values)
    # print(np.array(test_values).shape)
    # print(test_values)
    for i in range(window_size, len_values):
        X_test.append(test_values[i-window_size:i])
        y_test.append(test_values_y[i])
        # X_time.append(time[i])
    file_path = f'data/test/x_{colomn}.csv'
    # print(file_path)
    # print(np.array(X_test).shape)
    # if not os.path.exists(file_path):
    #     np.savetxt(file_path, np.array(X_test).squeeze(), delimiter=',')

    file_path = f'data/test/label.csv'
    # if not os.path.exists(file_path):
    #     np.savetxt(file_path, np.array(y_test).squeeze(), delimiter=',')
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    #print(f"X {X_test.shape}, y {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test
def get_data_binary_classification(colomn,df, window_size):
    a=["High", "Low", "Volume", "Open", "Close", "High_Low", "High_Volume", "High_Open", "High_Close", \
     "Low_Volume", "Low_Open", "Low_Close", "Volume_Open", "Volume_Close", "Open_Close", "High_Low_Volume_Open_Close", \
     "SMA_MACD_ROC_RSI_BB_CCI"]

    df_y=df[["Close"]]
    if colomn=="High":
        df=df[["High"]]
    elif colomn=="Low":
        df=df[["Low"]]
    elif colomn=="Volume":
        df=df[["Volume"]]
    elif colomn == "Open":
        df = df[["Open"]]
    elif colomn == "Close":
        df = df[["Close"]]
    elif colomn == "High_Low":
        df = df[["High","Low"]]
    elif colomn == "High_Volume":
        df = df[["High", "Volume"]]
    elif colomn == "High_Open":
        df = df[["High", "Open"]]
    elif colomn == "High_Close":
        df = df[["High", "Close"]]
    elif colomn == "Low_Volume":
        df = df[["Low", "Volume"]]
    elif colomn == "Low_Open":
        df = df[["Low", "Open"]]
    elif colomn == "Low_Close":
        df = df[["Low", "Close"]]
    elif colomn == "Volume_Open":
        df = df[["Volume", "Open"]]
    elif colomn == "Volume_Close":
        df = df[["Volume", "Close"]]
    elif colomn == "Open_Close":
        df = df[["Open", "Close"]]
    elif colomn == "High_Low_Volume_Open_Close":
        df = df[["High", "Low", "Volume", "Open", "Close"]]
    elif colomn == "Low_Close_BB":
        df = df[["Low", "Close","BB"]]
    elif colomn == "High_Low_Open_Close":
        df = df[["High", "Low", "Open", "Close"]]
        # print(df.values)
    elif colomn == "High_Low_Close":
        df = df[["High", "Low", "Close"]]
    elif colomn == "SMA_MACD_ROC_RSI_BB_CCI":
        df = df[["SMA", "MACD", "ROC", "RSI", "BB", "CCI"]]
    elif colomn == "BB":
        df = df[["BB"]]
    # "Close_BB", "Close_SMA", "Close_MACD", "Close_ROC", "Close_RSI", "Close_CCI"
    elif colomn == "Close_BB":
        df = df[["Close","BB"]]
    elif colomn == "Close_SMA":
        df = df[["Close","SMA"]]
    elif colomn == "Close_MACD":
        df = df[["Close","MACD"]]
    elif colomn == "Close_ROC":
        df = df[["Close","ROC"]]
    elif colomn == "Close_RSI":
        df = df[["Close","RSI"]]
    elif colomn == "Close_CCI":
        df = df[["Close","CCI"]]

    df_train = df[:- validation_size - test_size]
    # df_second_dataset=df[- (validation_size+test_size):]
    # csv_filename = 'historical_data_second_model.csv'
    # df_second_dataset.to_csv(csv_filename, index=False)
    df_validation = df[- validation_size - test_size:- test_size]
    df_test = df[- test_size:]
    scaler = MinMaxScaler()
    train_values = scaler.fit_transform(df_train.values)
    scaler_path = f'scalers/{interval}/{colomn}.bin'
    joblib.dump(scaler, scaler_path)
    val_values = scaler.transform(df_validation.values)
    test_values = scaler.transform(df_test.values)
    # print(np.array(df_test.values).shape)
    # exit()

    df_train_y = df_y[:- validation_size - test_size]
    df_validation_y = df_y[- validation_size - test_size:- test_size]
    df_test_y = df_y[- test_size:]
    scaler_y = MinMaxScaler()
    # train_values_y = scaler_y.fit_transform(df_train_y.values)
    # scaler_path = f'scalers/{interval}/label.bin'
    # joblib.dump(scaler_y, scaler_path)
    # val_values_y = scaler_y.transform(df_validation_y.values)
    # print(df_test_y.shape)
    # test_values_y = scaler_y.transform(df_test_y.values)
    # print(test_values_y.shape)

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    #train_values=np.array(train_values).squeeze()
    # if interval == "Hourly":
    #     min_change = 0.0003
    # if interval == "Daily":
    #     min_change = 0.0025
    len_values = len(train_values)
    # predicted_index=train_values.shape()[1]
    # print(predicted_index)
    # exit()
    for i in range(window_size, len_values):
        X_train.append(train_values[i-window_size:i,:])

        result=df_train_y.values[i]-df_train_y.values[i-1]
        # print(result)
        # exit()
        # if np.abs(result)<min_change:
        #     #natural
        #     y_train.append(2)
        if result >0:
            #buy
            y_train.append(1)
        else:
            # sell
            y_train.append(0)
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    #print(f"X {X_train.shape}, y {y_train.shape}")
    class_distribution = np.bincount(y_train)

    # If you want to label the classes for clarity
    class_labels = ["Sell", "Buy"]

    # Print the distribution
    for label, count in zip(class_labels, class_distribution):
        print(f"Class {label}: {count} samples")
    # print(f"X {X_test.shape}, y {y_test.shape}")
    # print(y_train)
    # exit()

    len_values = len(val_values)
    for i in range(window_size, len_values):
        X_val.append(val_values[i-window_size:i,:])
        result = df_validation_y.values[i] - df_validation_y.values[i - 1]

        if result > 0:
            # buy
            y_val.append(1)
        else:
            # sell
            y_val.append(0)
    X_val, y_val = np.asarray(X_val), np.asarray(y_val)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2]))

    len_values = len(test_values)
    for i in range(window_size, len_values):
        X_test.append(test_values[i-window_size:i])
        result = df_test_y.values[i] - df_test_y.values[i-1]

        if result > 0:
            # buy
            y_test.append(1)
        else:
            # sell
            y_test.append(0)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # y_train = to_categorical(y_train, num_classes=3)
    # y_val = to_categorical(y_val, num_classes=3)
    # y_test = to_categorical(y_test, num_classes=3)
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_data_with_predicted_binary_classification(colomn,df, window_size):
    a=["High", "Low", "Volume", "Open", "Close", "High_Low", "High_Volume", "High_Open", "High_Close", \
     "Low_Volume", "Low_Open", "Low_Close", "Volume_Open", "Volume_Close", "Open_Close", "High_Low_Volume_Open_Close", \
     "SMA_MACD_ROC_RSI_BB_CCI"]

    df_y=df[["Close"]]
    if colomn=="High":
        df=df[["High"]]
    elif colomn=="Low":
        df=df[["Low"]]
    elif colomn=="Volume":
        df=df[["Volume"]]
    elif colomn == "Open":
        df = df[["Open"]]
    elif colomn == "Close":
        df = df[["Close"]]
    elif colomn == "High_Low":
        df = df[["High","Low"]]
    elif colomn == "High_Volume":
        df = df[["High", "Volume"]]
    elif colomn == "High_Open":
        df = df[["High", "Open"]]
    elif colomn == "High_Close":
        df = df[["High", "Close"]]
    elif colomn == "Low_Volume":
        df = df[["Low", "Volume"]]
    elif colomn == "Low_Open":
        df = df[["Low", "Open"]]
    elif colomn == "Low_Close":
        df = df[["Low", "Close"]]
    elif colomn == "Volume_Open":
        df = df[["Volume", "Open"]]
    elif colomn == "Volume_Close":
        df = df[["Volume", "Close"]]
    elif colomn == "Open_Close":
        df = df[["Open", "Close"]]
    elif colomn == "High_Low_Volume_Open_Close":
        df = df[["High", "Low", "Volume", "Open", "Close"]]
    elif colomn == "Low_Close_BB":
        df = df[["Low", "Close","BB"]]
    elif colomn == "High_Low_Open_Close":
        df = df[["High", "Low", "Open", "Close"]]
        # print(df.values)
    elif colomn == "High_Low_Close":
        df = df[["High", "Low", "Close"]]
    elif colomn == "SMA_MACD_ROC_RSI_BB_CCI":
        df = df[["SMA", "MACD", "ROC", "RSI", "BB", "CCI"]]
    elif colomn == "BB":
        df = df[["BB"]]

    df_train = df[:- validation_size - test_size]
    # df_second_dataset=df[- (validation_size+test_size):]
    # csv_filename = 'historical_data_second_model.csv'
    # df_second_dataset.to_csv(csv_filename, index=False)
    df_validation = df[- validation_size - test_size:- test_size]
    df_test = df[- test_size:]
    scaler = MinMaxScaler()
    train_values = scaler.fit_transform(df_train.values)
    # train_values = df_train.values

    scaler_path = f'scalers/{interval}/{Problem}/{colomn}.bin'
    joblib.dump(scaler, scaler_path)
    val_values = scaler.transform(df_validation.values)
    test_values = scaler.transform(df_test.values)
    # print(np.array(df_test.values).shape)
    # exit()

    df_train_y = df_y[:- validation_size - test_size]
    df_validation_y = df_y[- validation_size - test_size:- test_size]
    df_test_y = df_y[- test_size:]
    scaler_y = MinMaxScaler()
    train_values_y = scaler_y.fit_transform(df_train_y.values)
    scaler_path = f'scalers/{interval}/{Problem}/label.bin'
    joblib.dump(scaler_y, scaler_path)
    val_values_y = scaler_y.transform(df_validation_y.values)
    # print(df_test_y.shape)
    test_values_y = scaler_y.transform(df_test_y.values)
    # print(test_values_y.shape)

    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    #train_values=np.array(train_values).squeeze()
    # if interval == "Hourly":
    #     min_change = 0.0003
    # if interval == "Daily":
    #     min_change = 0.0025
    len_values = len(train_values)
    predicted_index=train_values.shape[1]
    # print(predicted_index)
    # exit()
    for i in range(window_size, len_values):
        X_train.append(train_values[i-window_size:i,:predicted_index-1])


        result=df_train_y.values[i]-df_train_y.values[i-1]
        # print(result)
        # exit()
        # if np.abs(result)<min_change:
        #     #natural
        #     y_train.append(2)
        if result >0:
            #buy
            y_train.append(1)
        else:
            # sell
            y_train.append(0)
    print(train_values[:, predicted_index-1][:np.asarray(X_train).shape[0]].shape)
    print(np.asarray(X_train).shape)
    predicted_column = train_values[:, predicted_index-1][:np.asarray(X_train).shape[0]].reshape((-1, 1))

    # Tile the additional_column to repeat it 16 times
    predicted_column = np.tile(predicted_column, (1, window_size))
    # exit()
    X_train = np.dstack((X_train, predicted_column))

    print(X_train[0, :, :])
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    #print(f"X {X_train.shape}, y {y_train.shape}")
    class_distribution = np.bincount(y_train)

    # If you want to label the classes for clarity
    class_labels = ["Sell", "Buy"]

    # Print the distribution
    for label, count in zip(class_labels, class_distribution):
        print(f"Class {label}: {count} samples")
    # print(f"X {X_test.shape}, y {y_test.shape}")
    # print(y_train)
    # exit()

    len_values = len(val_values)
    for i in range(window_size, len_values):
        X_val.append(val_values[i-window_size:i,:])
        result = df_validation_y.values[i] - df_validation_y.values[i - 1]

        if result > 0:
            # buy
            y_val.append(1)
        else:
            # sell
            y_val.append(0)
    X_val, y_val = np.asarray(X_val), np.asarray(y_val)
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2]))
    #print(f"X {X_val.shape}, y {y_val.shape}")

    len_values = len(test_values)
    # print(np.array(test_values).shape)
    # print(test_values)

    for i in range(window_size, len_values):
        X_test.append(test_values[i-window_size:i])
        result = df_test_y.values[i] - df_test_y.values[i-1]

        if result > 0:
            # buy
            y_test.append(1)
        else:
            # sell
            y_test.append(0)
    file_path = f'data/test/x_{colomn}.csv'
    # print(file_path)
    # print(np.array(X_test).shape)
    # if not os.path.exists(file_path):
    #     np.savetxt(file_path, np.array(X_test).squeeze(), delimiter=',')

    file_path = f'data/test/label.csv'
    # if not os.path.exists(file_path):
    #     np.savetxt(file_path, np.array(y_test).squeeze(), delimiter=',')
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # y_train = to_categorical(y_train, num_classes=3)
    # y_val = to_categorical(y_val, num_classes=3)
    # y_test = to_categorical(y_test, num_classes=3)
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_train(values, window_size):
    X, y = [], []
    len_values = len(values)
    for i in range(window_size, len_values):
        X.append(values[i-window_size:i])
        y.append(values[i])
    X, y = np.asarray(X), np.asarray(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    print(f"X {X.shape}, y {y.shape}")
    return X, y

def get_val(values, window_size):
    X = []
    len_values = len(values)
    for i in range(window_size, len_values):
        X.append(values[i-window_size:i])
    X = np.asarray(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = values[-X.shape[0]:]
    print(f"X {X.shape}, y {y.shape}")
    return X, y

def get_train_multivariate(values, window_size):
    X, y = [], []
    x_values=values[:,1:]
    y_values=values[:,0]

    len_values = len(values)
    for i in range(window_size, len_values):
        X.append(x_values[i - window_size:i])
        y.append(y_values[i])
    X, y = np.asarray(X), np.asarray(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    print(f"X {X.shape}, y {y.shape}")
    return X, y


def get_val_multivariate(values, window_size):
    X, y = [], []
    x_values = values[:, 1:]
    y_values = values[:, 0]

    len_values = len(values)
    for i in range(window_size, len_values):
        X.append(x_values[i - window_size:i])
        y.append(y_values[i])
    X, y = np.asarray(X), np.asarray(y)

    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
    print(f"X {X.shape}, y {y.shape}")
    return X, y
