import matplotlib.pyplot as plt
import torch.nn.functional as F
from datetime import datetime
from sklearn.metrics import mean_squared_error as mse

def calculate_loss(pred, true, ratio=False, raw=None,columns=20):
    if ratio:
        if columns<10:
            loss = mse(pred[:,columns],true[:,columns]/raw[:,columns])
        else:
            loss = mse(pred,true/raw)
        return loss
    else:
        if columns<10:
            loss = mse(pred[:,columns]*raw[:,columns],true[:,columns])
        else:
            loss = mse(pred*raw,true)
        return loss

def plot_loss(result):
    plt.figure(figsize=(10, 5))
    plt.title("Result of LSTM")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(result[0][200:], label="train_loss")
    plt.plot(result[1][200:], label="test_loss")
    plt.legend()
    plt.show()

def plot_predict(result,train_data):
    tmp = list(result[4][-1][:, 0, 0]) + list(result[5][-1][:, 0, 0])
    plt.figure(figsize=(10, 5))
    dates = []
    for i in range(1999, 2022):
        for j in range(1, 13):
            dates.append(str(j) + '/' + str(i))
    xs = [datetime.strptime(d, '%m/%Y').date() for d in dates]
    xs = xs[204:]
    plt.title("Male ASMR")
    plt.plot(xs, tmp * train_data[-73:-1, 0], label="pred")
    plt.plot(xs, train_data[-72:, 0], label="truth")
    plt.plot(xs, train_data[-84:-12, 0], label="last year", linestyle='dashed')
    plt.legend()
    plt.show()
