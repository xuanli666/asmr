import pandas as pd
import numpy as np

def read_data(file_path ="data/tmp_all.csv"):
    data = pd.read_csv(file_path)
    train_data = data.iloc[:, 3::2]
    train_data = train_data.to_numpy()
    return train_data

def get_data(train_data,ratio=True,train_len=200,pred_len=1):
    x_train = []
    y_train = []
    if ratio:
        new_train_data = train_data.copy()
        for i in range(1,len(new_train_data)):
            new_train_data[i] = train_data[i] / train_data[i-1]
        new_train_data = new_train_data[1:]
        for i in range(len(new_train_data)-11-pred_len):
            x_train.append(new_train_data[i:i+12])
            y_train.append(new_train_data[i+12:i+12+pred_len])
#             y_train.append(new_train_data[i+12])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
    else:
        for i in range(len(train_data)-11-pred_len):
            x_train.append(train_data[i:i+12])
            y_train.append(train_data[i+12:i+12+pred_len])
#             y_train.append(train_data[i+12])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
    train_set = (x_train[:train_len],y_train[:train_len])
    test_set = (x_train[train_len:],y_train[train_len:])
    return train_set, test_set