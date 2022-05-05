import pandas as pd
from tst import Transformer
from lstm import netLSTM
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data import read_data, get_data
from utils import calculate_loss,plot_loss,plot_predict
from sklearn.metrics import mean_squared_error as mse

d_model = 12 # Lattent dim
q = 8 # Query size
v = 8 # Value size
h = 2 # Number of heads
N = 1 # Number of encoder and decoder to stack
attention_size = None#12 # Attention window size
dropout = 0.2 # Dropout rate
pe = 'regular' # Positional encoding
chunk_mode = None
d_input = 10  # From dataset
d_output = 10  # From dataset
train_data = read_data()
def train(model, train_set, test_set, opt_method='ensemble',model_name='transformer',column=0,pred_len=1,ratio=True):
    train_loss_trans = []
    test1_loss_trans =[]
    test2_loss_trans =[]
    train_pred_trans = []
    test1_pred_trans =[]
    test2_pred_trans =[]
    criterion = torch.nn.MSELoss()
    epochs = 1800
    for epoch in range(epochs):
        if epoch<100:
            lr_tmp =0.01
        elif 100<=epoch<300:
            lr_tmp = 0.003
        elif 300<=epoch<500:
            lr_tmp=0.001
        elif 500<=epoch < 800:
            lr_tmp = 3e-4
        else:
            lr_tmp=1e-4
        if not ratio:
            lr_tmp*=10
        opt = torch.optim.Adam(model.parameters(), lr=lr_tmp)
        input_,target = map(Variable, (torch.tensor(train_set[0]).float(), torch.tensor(train_set[1]).float()))
        input_ = input_.cuda()
        target = target.cuda()
        if model_name == 'lstm':
            pred = model(input_)[0]
            pred = pred.reshape(-1,pred_len,10)
#             print(pred.shape)
        elif model_name == 'linear':
            pred = model(input_.reshape(-1,12*10))
            pred = pred.reshape(-1,1,10)
        else:
            pred = model(input_)
            pred = pred[:,-pred_len:,:]
        if opt_method == 'alone':
            pred = pred[:,:,column]
            target = target[:,:,column]
#         print(target.shape)
        loss = criterion(pred, target)
        print("train_loss:",round(loss.item(),5))
        train_loss_trans.append(round(loss.item(),5))
        if epoch>=epochs-30:
            train_pred_trans.append(pred.cpu().detach().numpy())
        opt.zero_grad()
        loss.backward()
        opt.step()
        input_,target = map(Variable, (torch.tensor(test_set[0][:-24]).float(), torch.tensor(test_set[1][:-24]).float()))
        input_ = input_.cuda()
        target = target.cuda()
        if model_name == 'lstm':
            pred = model(input_)[0]
            pred = pred.reshape(-1,pred_len,10)
        elif model_name == 'linear':
            pred = model(input_.reshape(-1,12*10))
            pred = pred.reshape(-1,1,10)
        else:
            pred = model(input_)
            pred = pred[:,-pred_len:,:]
#             pred = pred[:,-1,:].reshape(-1,10)
        if opt_method == 'alone':
            pred = pred[:,:,column]
            target = target[:,:,column]
        loss = criterion(pred, target)
        print("train_loss:",round(loss.item(),5))
        test1_loss_trans.append(round(loss.item(),5))
        if epoch>=epochs-30:
            test1_pred_trans.append(pred.cpu().detach().numpy())
        input_,target = map(Variable, (torch.tensor(test_set[0][-24:]).float(), torch.tensor(test_set[1][-24:]).float()))
        input_ = input_.cuda()
        target = target.cuda()
        if model_name == 'lstm':
            pred = model(input_)[0]
            pred = pred.reshape(-1,pred_len,10)
        elif model_name == 'linear':
            pred = model(input_.reshape(-1,12*10))
            pred = pred.reshape(-1,1,10)
        else:
            pred = model(input_)
            pred = pred[:,-pred_len:,:]
#             pred = pred[:,-1,:].reshape(-1,10)
        if opt_method == 'alone':
            pred = pred[:,:,column]
            target = target[:,:,column]
        loss = criterion(pred, target)
        test2_loss_trans.append(round(loss.item(),5))
        if epoch>=epochs-30:
            test2_pred_trans.append(pred.cpu().detach().numpy())
    return train_loss_trans,test1_loss_trans,test2_loss_trans,train_pred_trans,test1_pred_trans,test2_pred_trans

def exper(ratio= True, train_len=191,opt_method='ensemble',model_name='transformer',column=0,pred_len=1):
    seed = 1
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    if model_name =='transformer':
        model = Transformer(d_input, d_model, d_output,
                    q, v, h, N, attention_size=attention_size,
                    dropout=dropout, chunk_mode=chunk_mode, pe=pe)
    elif model_name =='lstm':
        model = netLSTM(10,30,10,1,0.3,pred_len)
    elif model_name =='linear':
        model = nn.Linear(120,10)
    model = model.cuda()
    train_set,test_set = get_data(train_data,ratio=ratio,train_len=train_len,pred_len=pred_len)
    result = train(model, train_set, test_set, opt_method=opt_method,model_name=model_name,
                   column=column,pred_len=pred_len,ratio=ratio)
    return result

if __name__ == '__main__':
    result = exper(model_name='lstm')
    s = 0
    # for i in result[1][-30:]:
    #     s += i
    test_true = train_data[204:-24]
    raw = train_data[203:-25]
    for i in range(30):
        s += (calculate_loss(result[4][i].reshape(48, 10), test_true, ratio=False, raw=raw))
    s / 30
    print(s)
    plot_loss(result)
    plot_predict(result,train_data)
    # print(mse(train_data[13:204]/train_data[12:203],train_data[1:192]/train_data[:191]))